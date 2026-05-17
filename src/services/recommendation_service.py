from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from src.adapters.perfectdou_adapter import PerfectDouAdapter
from src.adapters.vision_adapter import RecognizedFrameState, parse_recognized_frame_state
from src.core.belief_sampler import BeliefSample, build_unknown_pool, sample_hidden_hands
from src.core.cards import CARD_RANKS, normalize_cards
from src.core.decision_engine import ActionEvaluation, evaluate_actions
from src.core.rule_engine import generate_legal_actions
from src.core.state_validator import ValidationResult, validate_visible_state
from src.core.visible_state import VisibleGameState


@dataclass(frozen=True, slots=True)
class RecommendationOptions:
    n_samples: int = 256
    rollout_per_action: int = 64
    mode: str = "perfectdou_monte_carlo"
    timeout_ms: int = 3000
    seed: int | None = 7

    @staticmethod
    def from_mapping(raw: Mapping[str, Any] | None) -> "RecommendationOptions":
        data = raw or {}
        seed_value = data.get("seed", 7)
        seed = None if seed_value is None else int(seed_value)
        return RecommendationOptions(
            n_samples=int(data.get("n_samples", 256)),
            rollout_per_action=int(data.get("rollout_per_action", 64)),
            mode=str(data.get("mode", "perfectdou_monte_carlo")),
            timeout_ms=int(data.get("timeout_ms", 3000)),
            seed=seed,
        )


class RecommendationService:
    def __init__(self, perfectdou_adapter: PerfectDouAdapter | None = None) -> None:
        self.perfectdou_adapter = perfectdou_adapter or PerfectDouAdapter()

    def dry_run(self, raw_state: Mapping[str, Any]) -> dict[str, object]:
        recognized = parse_recognized_frame_state(raw_state)
        visible_state = recognized.to_visible_state()
        validation = validate_visible_state(visible_state)
        return {
            "recognized_state": recognized.to_dict(),
            "visible_state": visible_state.to_dict(),
            "state_warnings": _merge_warnings(recognized.warnings, validation.warnings),
            "state_errors": list(validation.errors),
        }

    def recommend(
        self,
        raw_state: Mapping[str, Any] | RecognizedFrameState | VisibleGameState,
        options: RecommendationOptions | Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        resolved_options = options if isinstance(options, RecommendationOptions) else RecommendationOptions.from_mapping(options)
        recognized, visible_state, adapter_warnings = _coerce_state(raw_state)
        validation = validate_visible_state(visible_state)
        if not validation.ok:
            raise ValueError("; ".join(validation.errors))

        state_warnings = _merge_warnings(adapter_warnings, validation.warnings)
        if resolved_options.mode.startswith("perfectdou") and not self.perfectdou_adapter.available:
            state_warnings.append(f"PerfectDou 不可用: {self.perfectdou_adapter.unavailable_reason}; 已降级为启发式 score 模式")

        samples = self._sample_hands(visible_state, resolved_options, state_warnings)
        legal_actions = generate_legal_actions(visible_state)
        evaluations = evaluate_actions(
            visible_state,
            legal_actions,
            samples,
            rollout_per_action=resolved_options.rollout_per_action,
        )
        top_evaluations = evaluations[:3]

        if top_evaluations and top_evaluations[0].win_rate is None:
            state_warnings.append("win_rate 为 null，因为尚未完成 PerfectDou 全量 rollout；请使用 score 排序")

        belief_summary = self._belief_summary(visible_state, samples, state_warnings)
        response = {
            "best_action": list(top_evaluations[0].action) if top_evaluations else [],
            "top_actions": [_evaluation_to_dict(index + 1, evaluation) for index, evaluation in enumerate(top_evaluations)],
            "state_warnings": _dedupe(state_warnings),
            "belief_summary": belief_summary,
        }
        return response

    def _sample_hands(
        self,
        state: VisibleGameState,
        options: RecommendationOptions,
        warnings: list[str],
    ) -> list[BeliefSample]:
        try:
            samples = sample_hidden_hands(
                state,
                n_samples=options.n_samples,
                use_pass_constraints=True,
                seed=options.seed,
            )
        except ValueError as exc:
            warnings.append(f"隐藏牌采样已跳过: {exc}")
            samples = []
        if len(samples) < max(32, min(128, options.n_samples)):
            warnings.append(f"可用隐藏牌样本数只有 {len(samples)}")
        return samples

    def _belief_summary(
        self,
        state: VisibleGameState,
        samples: list[BeliefSample],
        warnings: list[str],
    ) -> dict[str, object]:
        try:
            unknown_pool = build_unknown_pool(state)
        except ValueError as exc:
            warnings.append(f"未知牌池摘要不可用: {exc}")
            unknown_pool = []
        counts = Counter(normalize_cards(unknown_pool))
        possible_bombs = [rank for rank in CARD_RANKS if rank not in {"X", "D"} and counts.get(rank, 0) >= 4]
        rocket_possible = counts.get("X", 0) > 0 and counts.get("D", 0) > 0
        return {
            "unknown_pool_size": len(unknown_pool),
            "possible_bombs": possible_bombs,
            "rocket_possible": rocket_possible,
            "samples_used": len(samples),
        }


def _coerce_state(
    raw_state: Mapping[str, Any] | RecognizedFrameState | VisibleGameState,
) -> tuple[RecognizedFrameState | None, VisibleGameState, list[str]]:
    if isinstance(raw_state, VisibleGameState):
        return None, raw_state, []
    if isinstance(raw_state, RecognizedFrameState):
        return raw_state, raw_state.to_visible_state(), list(raw_state.warnings)
    recognized = parse_recognized_frame_state(raw_state)
    return recognized, recognized.to_visible_state(), list(recognized.warnings)


def _evaluation_to_dict(rank: int, evaluation: ActionEvaluation) -> dict[str, object]:
    return {
        "rank": rank,
        "action": list(evaluation.action),
        "action_type": evaluation.action_type,
        "win_rate": evaluation.win_rate,
        "score": evaluation.score,
        "confidence": evaluation.confidence,
        "rollout_count": evaluation.rollout_count,
        "reasons": list(evaluation.reasons),
        "risks": list(evaluation.risks),
    }


def _merge_warnings(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return _dedupe(merged)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
