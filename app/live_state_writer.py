from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.cv.inference import CardDetector, CardDetectorConfig
from src.core.cards import CARD_LIMITS, assert_card_counts_within_deck
from src.core.rule_engine import RuleEngine, classify_action_cards
from src.domain.actions import ActionType, CardAction, CardDetection, PlayerSeat
from src.domain.cards import FULL_DECK_RANK_COUNTS
from src.io.screen_capture import ScreenCapture, ScreenCaptureConfig
from src.core.cards import from_domain_rank, normalize_cards, to_domain_rank


PREVIOUS_POSITION = {
    "landlord": "landlord_up",
    "landlord_down": "landlord",
    "landlord_up": "landlord_down",
}
NEXT_POSITION = {
    "landlord": "landlord_down",
    "landlord_down": "landlord_up",
    "landlord_up": "landlord",
}


@dataclass
class LiveActionMemory:
    last_non_pass_action: dict[str, object] | None = None
    retained_frames: int = 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    capture = ScreenCapture(ScreenCaptureConfig(window_title=args.window_title or config.window_title))
    detector = CardDetector(
        CardDetectorConfig(
            model_path=config.models.card_detector_path,
            confidence_threshold=args.confidence,
            nms_iou_threshold=config.cv.nms_iou_threshold,
            input_size=config.cv.input_size,
            class_names=config.cv.class_names,
        )
    )
    detector.load()
    if args.bring_window_front:
        capture.bring_window_to_front()
    memory = LiveActionMemory()
    try:
        while True:
            state = build_live_state(args, capture, detector, memory)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.once:
                print(json.dumps(state, ensure_ascii=False, indent=2))
                return
            time.sleep(max(0.2, args.interval))
    finally:
        capture.close()


def build_live_state(
    args: argparse.Namespace,
    capture: ScreenCapture,
    detector: CardDetector,
    memory: LiveActionMemory | None = None,
) -> dict[str, object]:
    frame = capture.capture()
    detections = detector.detect(frame)
    self_detections = _limit_detections(
        _detections_for_region(detections, PlayerSeat.SELF),
        _max_self_cards(args),
    )
    raw_table_detections = _table_candidate_detections(args, detections, frame)
    table_detections = _select_table_cluster(args, raw_table_detections, frame)
    self_hand = _cards_from_detections(self_detections)
    override_player, override_cards, override_force_lead, override_warning = _load_table_override(args)
    forced_table_cards = normalize_cards(args.last_table_cards or []) or ([] if override_force_lead else override_cards)
    forced_table_player = args.last_table_player if args.last_table_cards else override_player
    visual_table_cards = _cards_from_detections(table_detections)
    table_cards = [] if override_force_lead else (forced_table_cards or visual_table_cards)
    recognition_confidence = _mean_confidence(detections)
    self_hand_confidence = _mean_confidence(self_detections)
    table_confidence = 1.0 if forced_table_cards or override_force_lead else _mean_confidence(table_detections)
    detected_table_player = forced_table_player if forced_table_cards else _infer_table_player(args, table_detections, frame)
    table_shape_valid = _is_legal_table_shape(table_cards)
    table_cards_for_state = table_cards
    if table_cards and (table_confidence < args.min_table_confidence or not table_shape_valid):
        table_cards_for_state = []
    warnings: list[str] = []
    self_hand, table_cards_for_state, repair_warnings = _repair_visible_card_counts(self_hand, table_cards_for_state)
    warnings.extend(repair_warnings)
    table_cards = table_cards_for_state or table_cards
    table_shape_valid = _is_legal_table_shape(table_cards_for_state or table_cards)
    if frame.width == 0 or frame.height == 0:
        warnings.append("未捕获到 MuMu 窗口")
    if not detector.is_loaded():
        warnings.append("牌面识别模型未加载")
    if frame.width > 0 and frame.height > 0 and len(detections) < 5:
        warnings.append("检测到的牌面目标很少；请确认 MuMu 牌桌没有被 VS Code/Codex/终端窗口遮挡")
    if len(self_hand) == 0:
        warnings.append("未识别到当前手牌；如果处于叫地主/加倍/结算阶段，这是正常的")
    if len(table_cards) == 0 and not override_force_lead:
        warnings.append("未识别到桌面上一手牌")
    if override_warning:
        warnings.append(override_warning)
    if forced_table_cards:
        warnings.append(f"已使用强制上一手牌: {detected_table_player} {forced_table_cards}")
    if override_force_lead:
        warnings.append("已强制切换为自由出牌：不会沿用上一手，推荐里不会包含过")
    if table_cards and table_confidence < args.min_table_confidence:
        warnings.append(f"桌面牌识别置信度 {table_confidence:.2f} 低于 {args.min_table_confidence:.2f}，上一手牌可能误识别")
    if table_cards and not table_shape_valid:
        warnings.append(f"桌面牌 {table_cards} 不是合法斗地主牌型，已忽略上一手牌")
    deck_valid, deck_warning = _visible_cards_are_deck_valid(self_hand, table_cards_for_state)
    if not deck_valid:
        table_cards_for_state = []
        warnings.append(deck_warning)
    last_table_action, public_action_history, table_source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player=detected_table_player,
        candidate_cards=table_cards_for_state,
        candidate_confidence=table_confidence,
        manual=bool(forced_table_cards),
        force_lead=override_force_lead,
        warnings=warnings,
    )
    recommendation_ready, status_reason = _recommendation_readiness(
        args,
        self_hand,
        self_hand_confidence,
        table_cards,
        table_confidence,
        table_shape_valid,
        deck_valid,
    )
    if not recommendation_ready and status_reason:
        warnings.append(status_reason)

    num_cards_left = {
        "landlord": int(args.landlord_left),
        "landlord_down": int(args.landlord_down_left),
        "landlord_up": int(args.landlord_up_left),
    }
    num_cards_left[args.self_position] = len(self_hand)
    if last_table_action and not bool(last_table_action.get("is_pass", False)):
        action_player = str(last_table_action["player"])
        action_cards = list(last_table_action.get("cards", []))
        num_cards_left[action_player] = max(
            0,
            int(num_cards_left[action_player]) - len(action_cards),
        )

    state = {
        "self_hand": self_hand,
        "last_table_action": last_table_action,
        "public_action_history": public_action_history,
        "player_position": args.self_position,
        "acting_player": args.acting_player,
        "num_cards_left": num_cards_left,
        "landlord_public_cards": normalize_cards(args.landlord_public_cards),
        "recognition_confidence": recognition_confidence,
        "self_hand_confidence": self_hand_confidence,
        "table_confidence": table_confidence,
        "recommendation_ready": recommendation_ready,
        "status_reason": status_reason,
        "detected_table_cards": table_cards,
        "visual_table_cards": visual_table_cards,
        "detected_table_player": detected_table_player,
        "used_table_player": last_table_action.get("player") if last_table_action else None,
        "used_table_cards": list(last_table_action.get("cards", [])) if last_table_action else [],
        "used_table_action_type": _action_type(list(last_table_action.get("cards", []))) if last_table_action else "none",
        "used_table_source": table_source,
        "force_lead": override_force_lead,
        "memory_retained_frames": memory.retained_frames if memory else 0,
        "detected_counts": {
            "all": len(detections),
            "self": len(self_detections),
            "table": len(table_detections),
            "raw_table": len(raw_table_detections),
        },
        "writer_pid": os.getpid(),
        "writer_thresholds": {
            "min_recommend_confidence": args.min_recommend_confidence,
            "min_many_cards_confidence": args.min_many_cards_confidence,
            "min_table_confidence": args.min_table_confidence,
            "table_candidate_confidence": args.table_candidate_confidence,
            "new_action_min_confidence": args.new_action_min_confidence,
            "max_memory_retained_frames": args.max_memory_retained_frames,
            "prefer_last_table_player": args.prefer_last_table_player,
        },
        "frame_id": f"live:{frame.timestamp_ms}",
        "warnings": warnings,
    }
    if args.debug_image:
        _write_debug_image(args.debug_image, frame, detections)
    return state


def _cards_for_region(detections: tuple[CardDetection, ...], region: PlayerSeat) -> list[str]:
    return _cards_from_detections(_detections_for_region(detections, region))


def _detections_for_region(detections: tuple[CardDetection, ...], region: PlayerSeat) -> tuple[CardDetection, ...]:
    selected = [detection for detection in detections if detection.seat_region is region]
    selected = _dedupe_overlapping_detections(selected, iou_threshold=0.45)
    selected.sort(key=lambda detection: (detection.bbox.x, detection.bbox.y, -detection.confidence))
    return tuple(selected)


def _table_candidate_detections(
    args: argparse.Namespace,
    detections: tuple[CardDetection, ...],
    frame: object,
) -> tuple[CardDetection, ...]:
    selected = [
        detection
        for detection in detections
        if detection.seat_region is not PlayerSeat.SELF
        and _center_ratio(detection.bbox.y, detection.bbox.height, frame.height) >= args.table_min_y_ratio
        and _center_ratio(detection.bbox.y, detection.bbox.height, frame.height) <= args.table_max_y_ratio
    ]
    selected = _dedupe_overlapping_detections(selected, iou_threshold=0.45)
    selected.sort(key=lambda detection: (detection.bbox.x, detection.bbox.y, -detection.confidence))
    return tuple(selected)


def _select_table_cluster(
    args: argparse.Namespace,
    detections: tuple[CardDetection, ...],
    frame: object,
) -> tuple[CardDetection, ...]:
    candidates = [
        detection
        for detection in detections
        if detection.confidence >= args.table_candidate_confidence
        and _center_ratio(detection.bbox.x, detection.bbox.width, frame.width) >= args.table_min_x_ratio
        and _center_ratio(detection.bbox.x, detection.bbox.width, frame.width) <= args.table_max_x_ratio
        and _center_ratio(detection.bbox.y, detection.bbox.height, frame.height) >= args.table_min_y_ratio
        and _center_ratio(detection.bbox.y, detection.bbox.height, frame.height) <= args.table_max_y_ratio
    ]
    if not candidates:
        return ()

    row_clusters = _cluster_card_rows(candidates, frame.height * args.table_row_tolerance_ratio)
    best: tuple[float, tuple[CardDetection, ...]] | None = None
    for row in row_clusters:
        for group in _split_card_run(row, frame.width * args.table_gap_ratio):
            subset = _best_legal_subset(group)
            if not subset:
                subset = tuple(group)
            score = _table_group_score(subset, frame)
            if best is None or score > best[0]:
                best = (score, subset)
    if best is None:
        return ()
    selected = tuple(sorted(best[1], key=lambda detection: (detection.bbox.x, detection.bbox.y, -detection.confidence)))
    return selected


def _infer_table_player(args: argparse.Namespace, detections: tuple[CardDetection, ...], frame: object) -> str:
    if not detections:
        return args.last_table_player
    center_x = sum(detection.bbox.x + detection.bbox.width / 2.0 for detection in detections) / len(detections)
    ratio = _safe_ratio(center_x, frame.width)
    if ratio <= args.left_table_x_ratio:
        return args.left_player
    if ratio >= args.right_table_x_ratio:
        return args.right_player
    return args.last_table_player


def _resolve_last_table_action(
    *,
    args: argparse.Namespace,
    memory: LiveActionMemory | None,
    candidate_player: str,
    candidate_cards: list[str],
    candidate_confidence: float,
    manual: bool,
    warnings: list[str],
    force_lead: bool = False,
) -> tuple[dict[str, object] | None, list[dict[str, object]], str]:
    candidate = _make_action(candidate_player, candidate_cards) if candidate_cards else None
    if force_lead:
        if memory is not None:
            memory.last_non_pass_action = None
            memory.retained_frames = 0
        return None, [], "forced_lead"
    if memory is None or not args.remember_last_action:
        history = [candidate] if candidate else []
        return candidate, history, "detected"

    if manual and candidate is not None:
        memory.last_non_pass_action = candidate
        memory.retained_frames = 0
        return candidate, [candidate], "manual"

    remembered = memory.last_non_pass_action
    if candidate is not None:
        if remembered is None or _same_action(candidate, remembered) or _should_accept_new_action(
            args,
            memory,
            remembered,
            candidate,
            candidate_confidence,
        ):
            memory.last_non_pass_action = candidate
            memory.retained_frames = 0
            return candidate, [candidate], "detected"

        memory.retained_frames += 1
        if memory.retained_frames > args.max_memory_retained_frames:
            memory.last_non_pass_action = None
            warnings.append("最近一次非过牌已过期：长时间未识别到可信的新上一手，暂不沿用旧牌")
            return None, [], "expired"
        warnings.append(
            "已沿用最近一次非过牌：当前帧检测到的桌面牌没有形成新的有效应对，可能是有人不出或残留牌误识别"
        )
        return remembered, [remembered], "remembered_after_pass"

    if remembered is not None:
        memory.retained_frames += 1
        if memory.retained_frames > args.max_memory_retained_frames:
            memory.last_non_pass_action = None
            warnings.append("最近一次非过牌已过期：长时间未识别到新出牌，暂不沿用旧牌")
            return None, [], "expired"
        warnings.append("当前帧未识别到新出牌，已沿用最近一次非过牌")
        return remembered, [remembered], "remembered_no_new_cards"

    return None, [], "none"


def _make_action(player: str, cards: list[str]) -> dict[str, object]:
    return {"player": player, "cards": normalize_cards(cards), "is_pass": False}


def _same_action(left: dict[str, object], right: dict[str, object]) -> bool:
    return left.get("player") == right.get("player") and normalize_cards(left.get("cards") or []) == normalize_cards(
        right.get("cards") or []
    )


def _should_accept_new_action(
    args: argparse.Namespace,
    memory: LiveActionMemory,
    remembered: dict[str, object],
    candidate: dict[str, object],
    candidate_confidence: float,
) -> bool:
    candidate_cards = normalize_cards(candidate.get("cards") or [])
    remembered_cards = normalize_cards(remembered.get("cards") or [])
    if not candidate_cards:
        return False
    if candidate.get("player") != args.last_table_player and args.prefer_last_table_player:
        return False
    if candidate_confidence < args.new_action_min_confidence:
        return False
    if not remembered_cards:
        return True
    if candidate.get("player") == args.last_table_player and candidate_cards != remembered_cards:
        return True
    if memory.retained_frames >= args.same_player_refresh_after_frames and candidate_cards != remembered_cards:
        return True
    engine = RuleEngine()
    remembered_classified = engine.classify_action(_card_action_for(remembered_cards))
    candidate_action = _card_action_for(candidate_cards)
    candidate_classified = engine.classify_action(candidate_action)
    if candidate_classified.action_type is ActionType.INVALID:
        return False
    return engine.compare_actions(candidate_classified, remembered_classified)


def _card_action_for(cards: list[str]) -> CardAction:
    return CardAction(actor_seat=PlayerSeat.TABLE, ranks=tuple(to_domain_rank(card) for card in cards))


def _center_ratio(start: int, size: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (float(start) + float(size) / 2.0) / float(total)


def _cluster_card_rows(detections: list[CardDetection], y_tolerance: float) -> list[list[CardDetection]]:
    rows: list[list[CardDetection]] = []
    for detection in sorted(detections, key=lambda item: item.bbox.y + item.bbox.height / 2.0):
        center_y = detection.bbox.y + detection.bbox.height / 2.0
        matched_row: list[CardDetection] | None = None
        for row in rows:
            row_center = sum(item.bbox.y + item.bbox.height / 2.0 for item in row) / len(row)
            if abs(center_y - row_center) <= y_tolerance:
                matched_row = row
                break
        if matched_row is None:
            rows.append([detection])
        else:
            matched_row.append(detection)
    return rows


def _split_card_run(detections: list[CardDetection], max_gap: float) -> list[list[CardDetection]]:
    ordered = sorted(detections, key=lambda item: item.bbox.x)
    if not ordered:
        return []
    runs: list[list[CardDetection]] = [[ordered[0]]]
    for detection in ordered[1:]:
        previous = runs[-1][-1]
        previous_right = previous.bbox.x + previous.bbox.width
        gap = detection.bbox.x - previous_right
        if gap > max_gap:
            runs.append([detection])
        else:
            runs[-1].append(detection)
    return runs


def _best_legal_subset(detections: list[CardDetection]) -> tuple[CardDetection, ...]:
    if len(detections) > 10:
        cards = _cards_from_detections(tuple(detections))
        return tuple(detections) if _is_legal_table_shape(cards) else ()
    best: tuple[tuple[int, float], tuple[CardDetection, ...]] | None = None
    for size in range(len(detections), 0, -1):
        for subset in itertools.combinations(detections, size):
            cards = _cards_from_detections(tuple(subset))
            if not _is_legal_table_shape(cards):
                continue
            score = (len(cards), _mean_confidence(tuple(subset)))
            if best is None or score > best[0]:
                best = (score, tuple(subset))
        if best is not None and best[0][0] >= max(1, len(detections) - 1):
            break
    return best[1] if best is not None else ()


def _table_group_score(detections: tuple[CardDetection, ...], frame: object) -> float:
    if not detections:
        return 0.0
    card_count = len(_cards_from_detections(detections))
    mean_confidence = _mean_confidence(detections)
    center_x = sum(item.bbox.x + item.bbox.width / 2.0 for item in detections) / len(detections)
    center_y = sum(item.bbox.y + item.bbox.height / 2.0 for item in detections) / len(detections)
    x_bonus = 1.0 - min(1.0, abs(_safe_ratio(center_x, frame.width) - 0.5) * 2.0)
    y_bonus = 1.0 - min(1.0, abs(_safe_ratio(center_y, frame.height) - 0.36) * 2.0)
    legal_bonus = 2.0 if _is_legal_table_shape(_cards_from_detections(detections)) else 0.0
    return card_count * 10.0 + mean_confidence * 4.0 + x_bonus + y_bonus + legal_bonus


def _safe_ratio(value: float, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(value) / float(total)


def _dedupe_overlapping_detections(detections: list[CardDetection], *, iou_threshold: float) -> list[CardDetection]:
    selected: list[CardDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(_bbox_iou(detection, kept) < iou_threshold for kept in selected):
            selected.append(detection)
    return selected


def _limit_detections(detections: tuple[CardDetection, ...], max_cards: int) -> tuple[CardDetection, ...]:
    if max_cards <= 0 or len(detections) <= max_cards:
        return detections
    limited = sorted(detections, key=lambda item: item.confidence, reverse=True)[:max_cards]
    limited.sort(key=lambda detection: (detection.bbox.x, detection.bbox.y, -detection.confidence))
    return tuple(limited)


def _bbox_iou(left: CardDetection, right: CardDetection) -> float:
    left_box = left.bbox
    right_box = right.bbox
    x1 = max(left_box.x, right_box.x)
    y1 = max(left_box.y, right_box.y)
    x2 = min(left_box.x + left_box.width, right_box.x + right_box.width)
    y2 = min(left_box.y + left_box.height, right_box.y + right_box.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left_box.width) * max(0, left_box.height)
    right_area = max(0, right_box.width) * max(0, right_box.height)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _cards_from_detections(detections: tuple[CardDetection, ...]) -> list[str]:
    counts: Counter[str] = Counter()
    cards: list[str] = []
    for detection in detections:
        card = from_domain_rank(detection.card_rank)
        allowed = int(FULL_DECK_RANK_COUNTS[detection.card_rank])
        if counts[card] < allowed:
            counts[card] += 1
            cards.append(card)
    return normalize_cards(cards)


def _repair_visible_card_counts(self_hand: list[str], table_cards: list[str]) -> tuple[list[str], list[str], list[str]]:
    repaired_self = list(self_hand)
    repaired_table = list(table_cards)
    warnings: list[str] = []
    visible_counts = Counter(repaired_self + repaired_table)
    if visible_counts.get("X", 0) > CARD_LIMITS["X"] and visible_counts.get("D", 0) == 0:
        replaced = _replace_one_card(repaired_self, "X", "D")
        if not replaced:
            replaced = _replace_one_card(repaired_table, "X", "D")
        if replaced:
            warnings.append("视觉模型无法区分大小王：已将第二张 X 修正为 D")
    return normalize_cards(repaired_self), normalize_cards(repaired_table), warnings


def _replace_one_card(cards: list[str], old: str, new: str) -> bool:
    index = len(cards) - 1
    while index >= 0:
        if cards[index] == old:
            cards[index] = new
            return True
        index -= 1
    return False


def _visible_cards_are_deck_valid(self_hand: list[str], table_cards: list[str]) -> tuple[bool, str]:
    try:
        assert_card_counts_within_deck([*self_hand, *table_cards])
    except ValueError as exc:
        return False, f"识别结果超出一副牌限制，已暂停推荐: {exc}"
    return True, ""


def _load_table_override(args: argparse.Namespace) -> tuple[str, list[str], bool, str]:
    path = args.override_file
    if path is None or not path.exists():
        return args.last_table_player, [], False, ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return args.last_table_player, [], False, f"上一手覆盖文件读取失败: {exc}"
    if not bool(raw.get("enabled", False)):
        return args.last_table_player, [], False, ""
    updated_at_ms = _float_or_zero(raw.get("updated_at_ms"))
    ttl_seconds = float(raw.get("ttl_seconds", args.override_ttl_seconds))
    if updated_at_ms > 0 and time.time() * 1000.0 - updated_at_ms > ttl_seconds * 1000.0:
        return args.last_table_player, [], False, "上一手覆盖已过期，继续使用视觉识别"
    player = str(raw.get("last_table_player") or raw.get("player") or args.last_table_player)
    if player not in PREVIOUS_POSITION:
        return args.last_table_player, [], False, f"上一手覆盖玩家无效: {player}"
    force_lead = bool(raw.get("force_lead") or raw.get("new_trick") or raw.get("free_turn"))
    if force_lead:
        return player, [], True, "已读取自由出牌覆盖：忽略旧上一手"
    try:
        cards = normalize_cards(raw.get("last_table_cards") or raw.get("cards") or [])
    except ValueError as exc:
        return args.last_table_player, [], False, f"上一手覆盖牌无效: {exc}"
    if not cards:
        return args.last_table_player, [], False, ""
    return player, cards, False, f"已读取上一手覆盖: {player} {cards}"


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _recommendation_readiness(
    args: argparse.Namespace,
    self_hand: list[str],
    self_hand_confidence: float,
    table_cards: list[str] | None = None,
    table_confidence: float = 0.0,
    table_shape_valid: bool = True,
    deck_valid: bool = True,
) -> tuple[bool, str]:
    if not deck_valid:
        return False, "识别结果超出一副牌限制，暂不推荐"
    if len(self_hand) == 0:
        return False, "等待识别到当前手牌"
    min_self_confidence = _min_self_confidence(args, len(self_hand))
    if self_hand_confidence < min_self_confidence:
        return False, f"手牌识别置信度 {self_hand_confidence:.2f} 低于 {min_self_confidence:.2f}，暂不推荐"
    if table_cards and table_confidence < args.min_table_confidence:
        return False, f"桌面牌识别置信度 {table_confidence:.2f} 低于 {args.min_table_confidence:.2f}，暂不推荐"
    if table_cards and not table_shape_valid:
        return False, f"桌面牌 {table_cards} 不是合法斗地主牌型，暂不推荐"
    if args.acting_player != args.self_position:
        return False, f"当前 acting_player={args.acting_player}，不是 self_position={args.self_position} 的回合"
    return True, "可推荐"


def _min_self_confidence(args: argparse.Namespace, self_hand_count: int) -> float:
    if self_hand_count >= args.many_cards_threshold:
        return min(float(args.min_recommend_confidence), float(args.min_many_cards_confidence))
    return float(args.min_recommend_confidence)


def _max_self_cards(args: argparse.Namespace) -> int:
    if args.max_self_cards > 0:
        return int(args.max_self_cards)
    return 20 if args.self_position == "landlord" else 17


def _is_legal_table_shape(cards: list[str]) -> bool:
    if not cards:
        return True
    try:
        ranks = tuple(to_domain_rank(card) for card in cards)
    except ValueError:
        return False
    classified = RuleEngine().classify_action(CardAction(actor_seat=PlayerSeat.TABLE, ranks=ranks))
    return classified.action_type is not ActionType.INVALID


def _action_type(cards: list[str]) -> str:
    if not cards:
        return "none"
    try:
        return classify_action_cards(cards)
    except ValueError:
        return "invalid"


def _mean_confidence(detections: tuple[CardDetection, ...]) -> float:
    if not detections:
        return 0.0
    return round(sum(detection.confidence for detection in detections) / len(detections), 4)


def _write_debug_image(path: Path, frame: object, detections: tuple[CardDetection, ...]) -> None:
    if frame.width <= 0 or frame.height <= 0 or len(frame.pixels) != frame.width * frame.height * 3:
        return
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((frame.height, frame.width, 3))
    image = Image.fromarray(bgr[:, :, ::-1], mode="RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        PlayerSeat.SELF: "lime",
        PlayerSeat.TABLE: "yellow",
        PlayerSeat.LEFT_OPPONENT: "cyan",
        PlayerSeat.RIGHT_OPPONENT: "magenta",
    }
    for detection in detections:
        box = detection.bbox
        color = colors.get(detection.seat_region, "white")
        xyxy = (box.x, box.y, box.x + box.width, box.y + box.height)
        draw.rectangle(xyxy, outline=color, width=3)
        draw.text(
            (box.x, max(0, box.y - 14)),
            f"{from_domain_rank(detection.card_rank)} {detection.confidence:.2f} {detection.seat_region.value}",
            fill=color,
        )
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously write a v3 RecognizedFrameState JSON from live screen capture.")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--output", type=Path, default=Path("examples/live_test.json"))
    parser.add_argument("--window-title", default="")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.12)
    parser.add_argument("--min-recommend-confidence", type=float, default=0.45)
    parser.add_argument("--min-many-cards-confidence", type=float, default=0.28)
    parser.add_argument("--many-cards-threshold", type=int, default=8)
    parser.add_argument("--min-table-confidence", type=float, default=0.15)
    parser.add_argument("--table-candidate-confidence", type=float, default=0.05)
    parser.add_argument("--new-action-min-confidence", type=float, default=0.15)
    parser.add_argument("--same-player-refresh-after-frames", type=int, default=1)
    parser.add_argument("--max-memory-retained-frames", type=int, default=5)
    parser.add_argument("--table-min-x-ratio", type=float, default=0.28)
    parser.add_argument("--table-max-x-ratio", type=float, default=0.82)
    parser.add_argument("--table-min-y-ratio", type=float, default=0.20)
    parser.add_argument("--table-max-y-ratio", type=float, default=0.55)
    parser.add_argument("--left-table-x-ratio", type=float, default=0.45)
    parser.add_argument("--right-table-x-ratio", type=float, default=0.55)
    parser.add_argument("--table-row-tolerance-ratio", type=float, default=0.08)
    parser.add_argument("--table-gap-ratio", type=float, default=0.08)
    parser.add_argument("--max-self-cards", type=int, default=0)
    parser.add_argument("--self-position", choices=["landlord", "landlord_down", "landlord_up"], default="landlord")
    parser.add_argument("--acting-player", choices=["landlord", "landlord_down", "landlord_up"], default=None)
    parser.add_argument("--last-table-player", choices=["landlord", "landlord_down", "landlord_up"], default=None)
    parser.add_argument("--left-player", choices=["landlord", "landlord_down", "landlord_up"], default=None)
    parser.add_argument("--right-player", choices=["landlord", "landlord_down", "landlord_up"], default=None)
    parser.add_argument("--remember-last-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefer-last-table-player", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--landlord-left", type=int, default=20)
    parser.add_argument("--landlord-down-left", type=int, default=17)
    parser.add_argument("--landlord-up-left", type=int, default=17)
    parser.add_argument("--landlord-public-cards", nargs="*", default=[])
    parser.add_argument("--last-table-cards", nargs="*", default=[])
    parser.add_argument("--override-file", type=Path, default=Path("examples/live_override.json"))
    parser.add_argument("--override-ttl-seconds", type=float, default=30.0)
    parser.add_argument("--bring-window-front", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-image", type=Path, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.acting_player is None:
        args.acting_player = args.self_position
    if args.last_table_player is None:
        args.last_table_player = PREVIOUS_POSITION[args.self_position]
    if args.left_player is None:
        args.left_player = NEXT_POSITION[args.self_position]
    if args.right_player is None:
        args.right_player = PREVIOUS_POSITION[args.self_position]
    return args


if __name__ == "__main__":
    main()
