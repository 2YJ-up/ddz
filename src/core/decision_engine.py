from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Mapping

from src.core.belief_sampler import BeliefSample, build_unknown_pool
from src.core.cards import CARD_RANKS, count_cards, normalize_cards, subtract_cards, to_domain_rank
from src.core.rule_engine import RuleEngine
from src.core.visible_state import NEXT_POSITION, VisibleGameState
from src.domain.actions import ActionType, CardAction, PlayerSeat


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: list[str]
    action_type: str
    win_rate: float | None
    score: float
    confidence: Literal["low", "medium", "high"]
    rollout_count: int
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


def evaluate_actions(
    state: VisibleGameState,
    legal_actions: list[list[str]],
    samples: list[BeliefSample],
    rollout_per_action: int = 64,
    model_scores: Mapping[tuple[str, ...], float] | None = None,
    model_source: str | None = None,
) -> list[ActionEvaluation]:
    engine = RuleEngine()
    normalized_model_scores = _normalize_model_scores(model_scores or {})
    evaluations = [
        _evaluate_action(
            state,
            normalize_cards(action),
            samples,
            engine,
            rollout_per_action,
            model_score=normalized_model_scores.get(tuple(normalize_cards(action)), 0.0),
            model_source=model_source,
        )
        for action in legal_actions
    ]
    evaluations.sort(key=lambda item: item.score, reverse=True)
    return evaluations


def _evaluate_action(
    state: VisibleGameState,
    action: list[str],
    samples: list[BeliefSample],
    engine: RuleEngine,
    rollout_per_action: int,
    model_score: float = 0.0,
    model_source: str | None = None,
) -> ActionEvaluation:
    card_action = _to_card_action(action)
    classified = engine.classify_action(card_action)
    remaining = subtract_cards(state.self_hand, action)
    before_turns = _turn_count_estimate(state.self_hand)
    after_turns = _turn_count_estimate(remaining)
    reasons: list[str] = []
    risks: list[str] = []

    score = 10.0 + float(len(action)) * 1.6
    if classified.action_type is ActionType.PASS:
        score = 1.0
        reasons.append("不消耗控制牌")
    else:
        turn_reduction = max(0, before_turns - after_turns)
        score += float(turn_reduction) * 16.0
        if turn_reduction > 0:
            reasons.append(f"预计手数从 {before_turns} 降到 {after_turns}")
        if state.last_non_pass_action is not None:
            reasons.append("可压过当前牌")

    if not remaining:
        score += 120.0
        reasons.append("可直接出完")

    if model_score > 0.0:
        score += model_score
        source_text = model_source or "策略模型"
        reasons.append(f"{source_text} 模型评分支持")

    control_adjustment, control_reasons = _control_adjustment(state, action, classified.action_type, remaining)
    score += control_adjustment
    reasons.extend(control_reasons)

    break_penalty = _break_penalty(state.self_hand, action, classified.action_type)
    if break_penalty > 0:
        score -= break_penalty
        risks.append("拆散已有牌型结构")

    pressure_adjustment, pressure_risks = _opponent_pressure_adjustment(state, action, samples)
    score += pressure_adjustment
    risks.extend(pressure_risks)

    bomb_risks = _bomb_risks(state)
    if bomb_risks:
        risks.extend(bomb_risks)
        score -= min(12.0, float(len(bomb_risks)) * 2.0)

    if not reasons:
        reasons.append("可见信息启发式评分最高")
    risks.append("当前为 score 建议，尚未完成 PerfectDou 全量 rollout 胜率模拟")

    confidence = _confidence(state, samples, rollout_count=0)
    return ActionEvaluation(
        action=action,
        action_type=_action_type_name(classified.action_type),
        win_rate=None,
        score=round(max(0.001, score), 3),
        confidence=confidence,
        rollout_count=0,
        reasons=_dedupe(reasons),
        risks=_dedupe(risks),
    )


def _to_card_action(action: list[str]) -> CardAction:
    return CardAction(actor_seat=PlayerSeat.SELF, ranks=tuple(to_domain_rank(card) for card in action))


def _control_adjustment(
    state: VisibleGameState,
    action: list[str],
    action_type: ActionType,
    remaining: list[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    action_counts = count_cards(action)
    remaining_counts = count_cards(remaining)
    finishing = len(remaining) == 0

    control_cards = {"2", "X", "D"}
    spent_control = [card for card in control_cards if action_counts.get(card, 0) > 0]
    kept_control = [card for card in control_cards if remaining_counts.get(card, 0) > 0]
    if kept_control:
        score += float(len(kept_control)) * 2.5
        reasons.append("保留 2/王等控制牌")
    if spent_control and not finishing:
        score -= float(len(spent_control)) * 6.0
    if action_type in {ActionType.BOMB, ActionType.ROCKET} and not finishing:
        urgent = _minimum_opponent_count(state) <= 2
        if urgent:
            score += 8.0
            reasons.append("对手临近跑完，使用控制牌压制")
        else:
            score -= 16.0
    return score, reasons


def _break_penalty(hand: list[str], action: list[str], action_type: ActionType) -> float:
    if action_type in {
        ActionType.STRAIGHT,
        ActionType.PAIR_STRAIGHT,
        ActionType.TRIPLE_STRAIGHT,
        ActionType.AIRPLANE_WITH_SINGLES,
        ActionType.AIRPLANE_WITH_PAIRS,
    }:
        return 0.0
    hand_counts = count_cards(hand)
    action_counts = count_cards(action)
    penalty = 0.0
    for card, used in action_counts.items():
        owned = hand_counts.get(card, 0)
        if owned >= 4 and used < 4:
            penalty += 18.0
        elif owned == 3 and used < 3:
            penalty += 7.0
        elif owned == 2 and used == 1:
            penalty += 3.5
    if _sequence_savings(hand_counts) > _sequence_savings(count_cards(subtract_cards(hand, action))):
        penalty += 5.0
    return penalty


def _opponent_pressure_adjustment(
    state: VisibleGameState,
    action: list[str],
    samples: list[BeliefSample],
) -> tuple[float, list[str]]:
    score = 0.0
    risks: list[str] = []
    next_player = NEXT_POSITION.get(state.self_position)
    if next_player is not None and state.num_cards_left.get(next_player, 99) <= 2:
        risks.append(f"{next_player} 剩余 {state.num_cards_left[next_player]} 张")
        if action:
            score += 5.0
        else:
            score -= 25.0

    if samples:
        risky_weight = 0.0
        total_weight = sum(sample.weight for sample in samples)
        for sample in samples:
            sample_has_one_move_risk = any(
                len(cards) <= 3 and _can_finish_in_one_move(cards) for cards in sample.hands.values()
            )
            if sample_has_one_move_risk:
                risky_weight += sample.weight
        if total_weight > 0:
            risk_ratio = risky_weight / total_weight
            if risk_ratio > 0.25 and action:
                risks.append("采样显示对手存在一手跑完风险")
                score -= risk_ratio * 4.0
    return score, risks


def _bomb_risks(state: VisibleGameState) -> list[str]:
    unknown_counts = count_cards(build_unknown_pool(state))
    risks: list[str] = []
    possible_bombs = [rank for rank in CARD_RANKS if rank not in {"X", "D"} and unknown_counts.get(rank, 0) >= 4]
    if possible_bombs:
        risks.append("外面可能有炸弹: " + ",".join(possible_bombs))
    if unknown_counts.get("X", 0) > 0 and unknown_counts.get("D", 0) > 0:
        risks.append("外面可能还有王炸")
    return risks


def _minimum_opponent_count(state: VisibleGameState) -> int:
    counts = [state.num_cards_left[player] for player in state.num_cards_left if player != state.self_position]
    return min(counts) if counts else 99


def _confidence(
    state: VisibleGameState,
    samples: list[BeliefSample],
    *,
    rollout_count: int,
) -> Literal["low", "medium", "high"]:
    if state.recognition_confidence < 0.85 or len(samples) < 32:
        return "low"
    if rollout_count > 0 and len(samples) >= 128:
        return "high"
    return "medium"


def _normalize_model_scores(model_scores: Mapping[tuple[str, ...], float]) -> dict[tuple[str, ...], float]:
    finite_scores = [float(value) for value in model_scores.values() if _is_finite(value)]
    normalized: dict[tuple[str, ...], float] = {}
    if finite_scores:
        min_score = min(finite_scores)
        max_score = max(finite_scores)
        span = max_score - min_score
        for action, value in model_scores.items():
            if _is_finite(value):
                if span <= 1.0e-9:
                    normalized[action] = 20.0
                else:
                    normalized[action] = 5.0 + 45.0 * ((float(value) - min_score) / span)
    return normalized


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _turn_count_estimate(cards: list[str] | Mapping[str, int]) -> int:
    counts = count_cards(cards) if not isinstance(cards, Mapping) else Counter({str(key): int(value) for key, value in cards.items()})
    total = sum(counts.values())
    if total == 0:
        return 0
    groups = sum(1 for amount in counts.values() if amount > 0)
    return max(1, groups - _sequence_savings(counts))


def _sequence_savings(counts: Mapping[str, int]) -> int:
    best = 0
    current = 0
    for rank in ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"):
        if counts.get(rank, 0) >= 1:
            current += 1
        else:
            best = max(best, current)
            current = 0
    best = max(best, current)
    return best - 1 if best >= 5 else 0


def _can_finish_in_one_move(cards: list[str]) -> bool:
    if not cards:
        return False
    engine = RuleEngine()
    classified = engine.classify_action(_to_card_action(cards))
    return classified.action_type not in {ActionType.INVALID, ActionType.PASS}


def _action_type_name(action_type: ActionType) -> str:
    mapping = {
        ActionType.PASS: "pass",
        ActionType.SINGLE: "single",
        ActionType.PAIR: "pair",
        ActionType.TRIPLE: "trio",
        ActionType.TRIPLE_WITH_SINGLE: "trio_with_single",
        ActionType.TRIPLE_WITH_PAIR: "trio_with_pair",
        ActionType.STRAIGHT: "straight",
        ActionType.PAIR_STRAIGHT: "pair_straight",
        ActionType.TRIPLE_STRAIGHT: "trio_straight",
        ActionType.AIRPLANE_WITH_SINGLES: "airplane_with_singles",
        ActionType.AIRPLANE_WITH_PAIRS: "airplane_with_pairs",
        ActionType.FOUR_WITH_TWO_SINGLES: "four_with_two_singles",
        ActionType.FOUR_WITH_TWO_PAIRS: "four_with_two_pairs",
        ActionType.BOMB: "bomb",
        ActionType.ROCKET: "rocket",
        ActionType.INVALID: "invalid",
        ActionType.UNKNOWN: "unknown",
    }
    return mapping[action_type]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
