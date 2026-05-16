from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.core.rule_engine import RuleEngine
from src.domain.actions import ActionRecommendation, CardAction, GameStateView
from src.domain.cards import CardRank


@dataclass(frozen=True, slots=True)
class RlForwardConfig:
    model_path: Path
    top_n: int


class RlForwardService:
    def __init__(self, config: RlForwardConfig, rule_engine: RuleEngine) -> None:
        self.config = config
        self.rule_engine = rule_engine
        self._model_loaded = False

    def load(self) -> None:
        self._model_loaded = self.config.model_path.exists()

    def is_loaded(self) -> bool:
        result = self._model_loaded
        return result

    def build_state_tensor(self, state_view: GameStateView) -> tuple[int, ...]:
        tensor = state_view.state_matrix
        return tensor

    def recommend(
        self,
        state_view: GameStateView,
        hand_rank_counts: Mapping[CardRank, int] | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        effective_hand_counts: Mapping[CardRank, int]
        if hand_rank_counts is None:
            effective_hand_counts = {}
        else:
            effective_hand_counts = hand_rank_counts

        legal_actions = self.rule_engine.enumerate_legal_actions(
            hand_counts=effective_hand_counts,
            actor_seat=state_view.current_turn,
            previous_action=state_view.current_trick_action,
        )
        scored_actions = self._score_actions(tuple(legal_actions))
        limited_actions = scored_actions[: self.config.top_n]
        recommendations = self._to_recommendations(limited_actions)
        return recommendations

    def _score_actions(self, actions: tuple[CardAction, ...]) -> tuple[tuple[CardAction, float], ...]:
        scored: list[tuple[CardAction, float]] = []
        for action in actions:
            classified = self.rule_engine.classify_action(action)
            base_score = float(action.card_count())
            if classified.primary_rank is not None:
                base_score += float(int(classified.primary_rank)) / 100.0
            if classified.action_type.value == "pass":
                base_score = 0.01
            scored.append((action, base_score))
        result = tuple(sorted(scored, key=lambda item: item[1], reverse=True))
        return result

    def _to_recommendations(
        self,
        scored_actions: tuple[tuple[CardAction, float], ...],
    ) -> tuple[ActionRecommendation, ...]:
        total_score = sum(score for _, score in scored_actions)
        recommendations: list[ActionRecommendation] = []
        for action, score in scored_actions:
            probability = 0.0
            if total_score > 0.0:
                probability = score / total_score
            expected_win_rate = min(0.99, max(0.01, probability))
            recommendations.append(
                ActionRecommendation(
                    action=action,
                    probability=probability,
                    expected_win_rate=expected_win_rate,
                    reason_code="heuristic_fallback",
                )
            )
        result = tuple(recommendations)
        return result
