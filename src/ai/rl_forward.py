from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.core.rule_engine import RuleEngine
from src.domain.actions import ActionRecommendation, ActionType, CardAction, GameStateView, PlayerSeat
from src.domain.cards import CardRank


@dataclass(frozen=True, slots=True)
class RlForwardConfig:
    model_path: Path
    top_n: int
    input_name: str = ""
    output_name: str = ""


class RlForwardService:
    def __init__(self, config: RlForwardConfig, rule_engine: RuleEngine) -> None:
        self.config = config
        self.rule_engine = rule_engine
        self._session: Any = None
        self._input_name = config.input_name
        self._output_name = config.output_name
        self._model_loaded = False

    def load(self) -> None:
        self._session = None
        self._model_loaded = False
        if self.config.model_path.exists():
            self._load_onnx_session()

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
            if state_view.current_turn is PlayerSeat.SELF:
                effective_hand_counts = state_view.self_rank_counts
            else:
                effective_hand_counts = {}
        else:
            effective_hand_counts = hand_rank_counts

        legal_actions = self.rule_engine.enumerate_legal_actions(
            hand_counts=effective_hand_counts,
            actor_seat=state_view.current_turn,
            previous_action=state_view.current_trick_action,
        )
        policy_scores = self._run_policy(state_view)
        if policy_scores.size > 0:
            scored_actions = self._score_actions_with_policy(tuple(legal_actions), policy_scores)
        else:
            scored_actions = self._score_actions_heuristically(tuple(legal_actions))
        limited_actions = scored_actions[: self.config.top_n]
        reason_code = "onnx_policy" if policy_scores.size > 0 else "heuristic_fallback"
        recommendations = self._to_recommendations(limited_actions, reason_code)
        return recommendations

    def _load_onnx_session(self) -> None:
        try:
            ort_module = importlib.import_module("onnxruntime")
            self._session = ort_module.InferenceSession(str(self.config.model_path), providers=["CPUExecutionProvider"])
            if self._input_name == "":
                self._input_name = self._session.get_inputs()[0].name
            if self._output_name == "":
                self._output_name = self._session.get_outputs()[0].name
            self._model_loaded = True
        except (ImportError, OSError, RuntimeError, ValueError):
            self._session = None
            self._model_loaded = False

    def _run_policy(self, state_view: GameStateView) -> np.ndarray:
        scores = np.array([], dtype=np.float32)
        if self._model_loaded and self._session is not None:
            tensor = np.asarray(state_view.state_matrix, dtype=np.float32).reshape(1, -1)
            try:
                outputs = self._session.run([self._output_name], {self._input_name: tensor})
                scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            except (RuntimeError, ValueError):
                scores = np.array([], dtype=np.float32)
        return scores

    def _score_actions_with_policy(
        self,
        actions: tuple[CardAction, ...],
        policy_scores: np.ndarray,
    ) -> tuple[tuple[CardAction, float], ...]:
        scored: list[tuple[CardAction, float]] = []
        for action in actions:
            index = self._policy_index_for_action(action, policy_scores.size)
            model_score = float(policy_scores[index]) if policy_scores.size > 0 else 0.0
            score = max(0.0, model_score)
            if score == 0.0:
                score = 0.0001
            scored.append((action, score))
        result = tuple(sorted(scored, key=lambda item: item[1], reverse=True))
        return result

    def _score_actions_heuristically(self, actions: tuple[CardAction, ...]) -> tuple[tuple[CardAction, float], ...]:
        scored: list[tuple[CardAction, float]] = []
        for action in actions:
            classified = self.rule_engine.classify_action(action)
            base_score = float(action.card_count())
            if classified.primary_rank is not None:
                base_score += float(int(classified.primary_rank)) / 100.0
            if classified.action_type is ActionType.PASS:
                base_score = 0.01
            scored.append((action, base_score))
        result = tuple(sorted(scored, key=lambda item: item[1], reverse=True))
        return result

    def _to_recommendations(
        self,
        scored_actions: tuple[tuple[CardAction, float], ...],
        reason_code: str,
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
                    reason_code=reason_code,
                )
            )
        result = tuple(recommendations)
        return result

    def _policy_index_for_action(self, action: CardAction, policy_size: int) -> int:
        classified = self.rule_engine.classify_action(action)
        type_index = self._action_type_index(classified.action_type)
        rank_index = 0
        if classified.primary_rank is not None:
            rank_index = max(0, int(classified.primary_rank) - int(CardRank.THREE))
        raw_index = type_index * 15 + rank_index
        index = raw_index % max(1, policy_size)
        return index

    def _action_type_index(self, action_type: ActionType) -> int:
        ordered_types = (
            ActionType.PASS,
            ActionType.SINGLE,
            ActionType.PAIR,
            ActionType.TRIPLE,
            ActionType.TRIPLE_WITH_SINGLE,
            ActionType.TRIPLE_WITH_PAIR,
            ActionType.STRAIGHT,
            ActionType.PAIR_STRAIGHT,
            ActionType.TRIPLE_STRAIGHT,
            ActionType.AIRPLANE_WITH_SINGLES,
            ActionType.AIRPLANE_WITH_PAIRS,
            ActionType.FOUR_WITH_TWO_SINGLES,
            ActionType.FOUR_WITH_TWO_PAIRS,
            ActionType.BOMB,
            ActionType.ROCKET,
        )
        index = 0
        cursor = 0
        while cursor < len(ordered_types):
            if ordered_types[cursor] is action_type:
                index = cursor
            cursor += 1
        return index
