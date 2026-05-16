from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.core.rule_engine import RuleEngine
from src.domain.actions import ActionRecommendation, ActionType, CardAction, GameStateView, PlayerSeat
from src.domain.cards import CardRank, FULL_DECK_RANK_COUNTS


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
        self._expected_input_size = 0
        self._model_loaded = False
        self._dimension_warning_emitted = False
        self._turn_count_cache: dict[tuple[int, ...], int] = {}

    def load(self) -> None:
        self._session = None
        self._model_loaded = False
        if self.config.model_path.exists():
            self._load_onnx_session()

    def is_loaded(self) -> bool:
        result = self._model_loaded
        return result

    def expected_input_size(self) -> int:
        result = self._expected_input_size
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
            effective_hand_counts = state_view.self_rank_counts
        else:
            effective_hand_counts = hand_rank_counts
        self._turn_count_cache = {}

        legal_actions = self.rule_engine.enumerate_legal_actions(
            hand_counts=effective_hand_counts,
            actor_seat=PlayerSeat.SELF,
            previous_action=state_view.current_trick_action,
        )
        policy_scores: np.ndarray | None = None
        scored_actions: tuple[tuple[CardAction, float], ...] = ()
        if len(legal_actions) > 0:
            policy_scores = self._run_policy(state_view)
            if policy_scores is not None and policy_scores.size > 0:
                scored_actions = self._score_actions_with_policy(tuple(legal_actions), policy_scores)
            else:
                scored_actions = self._score_actions_heuristically(tuple(legal_actions), state_view)
        limited_actions = scored_actions[: self.config.top_n]
        reason_code = "onnx_policy" if policy_scores is not None and policy_scores.size > 0 else "heuristic_fallback"
        recommendations = self._to_recommendations(limited_actions, reason_code, state_view)
        return recommendations

    def _load_onnx_session(self) -> None:
        try:
            ort_module = importlib.import_module("onnxruntime")
            self._session = ort_module.InferenceSession(str(self.config.model_path), providers=["CPUExecutionProvider"])
            input_meta = self._session.get_inputs()[0]
            if self._input_name == "":
                self._input_name = input_meta.name
            self._expected_input_size = self._resolve_expected_input_size(input_meta.shape)
            if self._output_name == "":
                self._output_name = self._session.get_outputs()[0].name
            self._model_loaded = True
        except (ImportError, OSError, RuntimeError, ValueError):
            self._session = None
            self._model_loaded = False

    def _run_policy(self, state_view: GameStateView) -> np.ndarray | None:
        scores: np.ndarray | None = np.array([], dtype=np.float32)
        if self._model_loaded and self._session is not None:
            if self._policy_input_matches_state(state_view):
                tensor = self._build_policy_tensor(state_view)
                try:
                    outputs = self._session.run([self._output_name], {self._input_name: tensor})
                    scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
                except Exception:
                    self._warn_policy_dimension_mismatch()
                    scores = None
            else:
                self._warn_policy_dimension_mismatch()
                scores = None
        return scores

    def _build_policy_tensor(self, state_view: GameStateView) -> np.ndarray:
        base_tensor = np.asarray(state_view.state_matrix, dtype=np.float32).reshape(-1)
        tensor = base_tensor.reshape((1, int(base_tensor.size)))
        return tensor

    def _policy_input_matches_state(self, state_view: GameStateView) -> bool:
        state_size = len(state_view.state_matrix)
        matches = self._expected_input_size <= 0 or self._expected_input_size == state_size
        return matches

    def _warn_policy_dimension_mismatch(self) -> None:
        if not self._dimension_warning_emitted:
            print("Model dimension mismatch, falling back to heuristics.")
            self._dimension_warning_emitted = True

    def _resolve_expected_input_size(self, input_shape: tuple | list) -> int:
        expected_size = 0
        if len(input_shape) >= 2 and isinstance(input_shape[1], int):
            expected_size = int(input_shape[1])
        return expected_size

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

    def _score_actions_heuristically(
        self,
        actions: tuple[CardAction, ...],
        state_view: GameStateView | None = None,
    ) -> tuple[tuple[CardAction, float], ...]:
        scored: list[tuple[CardAction, float]] = []
        for action in actions:
            classified = self.rule_engine.classify_action(action)
            base_score = self._base_heuristic_score(action, state_view)
            if classified.primary_rank is not None:
                base_score += self._primary_rank_adjustment(classified.action_type, classified.primary_rank)
            if classified.action_type is ActionType.PASS:
                base_score = self._pass_score(state_view)
            scored.append((action, base_score))
        result = tuple(sorted(scored, key=lambda item: item[1], reverse=True))
        return result

    def _base_heuristic_score(self, action: CardAction, state_view: GameStateView | None) -> float:
        score = float(action.card_count()) * 1.2
        if state_view is not None:
            remaining_counts = self._remaining_counts_after_action(state_view.self_rank_counts, action)
            remaining_card_count = sum(remaining_counts.values())
            before_turns = self._turn_count_estimate(state_view.self_rank_counts)
            after_turns = self._turn_count_estimate(remaining_counts)
            turn_reduction = max(0, before_turns - after_turns)
            score += float(turn_reduction) * 12.0
            score -= float(after_turns) * 1.6
            score += max(0.0, 18.0 - float(remaining_card_count)) * 0.35
            score += self._control_score(action, state_view)
            score -= self._break_penalty(action, state_view.self_rank_counts)
            if remaining_card_count == 0:
                score += 100.0
            score += self._endgame_pressure_bonus(action, state_view)

        classified = self.rule_engine.classify_action(action)
        if classified.action_type is ActionType.BOMB:
            score += self._bomb_bonus(action, state_view)
        elif classified.action_type is ActionType.ROCKET:
            score += self._rocket_bonus(action, state_view)
        return max(0.001, score)

    def _remaining_counts_after_action(
        self,
        hand_counts: Mapping[CardRank, int],
        action: CardAction,
    ) -> dict[CardRank, int]:
        remaining = {rank: int(count) for rank, count in hand_counts.items() if count > 0}
        for rank in action.ranks:
            current_count = remaining.get(rank, 0)
            next_count = max(0, current_count - 1)
            if next_count > 0:
                remaining[rank] = next_count
            else:
                remaining.pop(rank, None)
        return remaining

    def _hand_complexity(self, hand_counts: Mapping[CardRank, int]) -> float:
        singles = 0
        pairs = 0
        triples = 0
        bombs = 0
        for _, count in hand_counts.items():
            if count == 1:
                singles += 1
            elif count == 2:
                pairs += 1
            elif count == 3:
                triples += 1
            elif count >= 4:
                bombs += 1
        complexity = float(singles) * 1.4 + float(pairs) * 0.8 + float(triples) * 0.45 - float(bombs) * 0.7
        return max(0.0, complexity)

    def _control_score(self, action: CardAction, state_view: GameStateView) -> float:
        classified = self.rule_engine.classify_action(action)
        score = 0.0
        if classified.action_type is ActionType.ROCKET:
            score = 18.0
        elif classified.action_type is ActionType.BOMB:
            score = 10.0
            score -= self._higher_bomb_or_rocket_risk(classified, state_view) * 1.5
        elif classified.primary_rank is not None:
            higher_options = self._higher_unknown_response_count(classified, state_view)
            if higher_options == 0:
                score += 6.0
            else:
                score -= min(6.0, float(higher_options) * 0.8)
            score -= min(5.0, float(self._unseen_bomb_risk_count(state_view)) * 0.7)
            if state_view.current_trick_action is not None:
                score += self._response_efficiency_bonus(classified, state_view.current_trick_action)
        return score

    def _higher_unknown_response_count(
        self,
        classified: object,
        state_view: GameStateView,
    ) -> int:
        higher_options = 0
        action_type = getattr(classified, "action_type", ActionType.INVALID)
        primary_rank = getattr(classified, "primary_rank", None)
        required_count = self._required_same_rank_count(action_type)
        if primary_rank is not None and required_count > 0:
            for rank, count in state_view.known_unseen_counts.items():
                if int(rank) > int(primary_rank) and count >= required_count:
                    higher_options += 1
        return higher_options

    def _higher_bomb_or_rocket_risk(self, classified: object, state_view: GameStateView) -> int:
        risk = 0
        primary_rank = getattr(classified, "primary_rank", None)
        if primary_rank is not None:
            for rank, count in state_view.known_unseen_counts.items():
                if rank not in {CardRank.SMALL_JOKER, CardRank.BIG_JOKER} and int(rank) > int(primary_rank) and count >= 4:
                    risk += 1
        has_small_joker = state_view.known_unseen_counts.get(CardRank.SMALL_JOKER, 0) > 0
        has_big_joker = state_view.known_unseen_counts.get(CardRank.BIG_JOKER, 0) > 0
        if has_small_joker and has_big_joker:
            risk += 1
        return risk

    def _unseen_bomb_risk_count(self, state_view: GameStateView) -> int:
        risk = 0
        for rank, count in state_view.known_unseen_counts.items():
            if rank not in {CardRank.SMALL_JOKER, CardRank.BIG_JOKER} and count >= 4:
                risk += 1
        has_small_joker = state_view.known_unseen_counts.get(CardRank.SMALL_JOKER, 0) > 0
        has_big_joker = state_view.known_unseen_counts.get(CardRank.BIG_JOKER, 0) > 0
        if has_small_joker and has_big_joker:
            risk += 1
        return risk

    def _required_same_rank_count(self, action_type: ActionType) -> int:
        required_count = 0
        if action_type is ActionType.SINGLE:
            required_count = 1
        elif action_type is ActionType.PAIR:
            required_count = 2
        elif action_type is ActionType.TRIPLE:
            required_count = 3
        elif action_type is ActionType.TRIPLE_WITH_SINGLE:
            required_count = 3
        elif action_type is ActionType.TRIPLE_WITH_PAIR:
            required_count = 3
        elif action_type is ActionType.BOMB:
            required_count = 4
        return required_count

    def _response_efficiency_bonus(self, classified: object, previous_action: object) -> float:
        bonus = 0.0
        action_type = getattr(classified, "action_type", ActionType.INVALID)
        primary_rank = getattr(classified, "primary_rank", None)
        previous_type = getattr(previous_action, "action_type", ActionType.INVALID)
        previous_rank = getattr(previous_action, "primary_rank", None)
        if primary_rank is not None and previous_rank is not None and action_type is previous_type:
            rank_gap = int(primary_rank) - int(previous_rank)
            bonus += max(0.0, 4.0 - float(rank_gap) * 0.6)
        return bonus

    def _break_penalty(self, action: CardAction, hand_counts: Mapping[CardRank, int]) -> float:
        penalty = 0.0
        action_counts = self._action_rank_counts(action)
        for rank, used_count in action_counts.items():
            owned_count = int(hand_counts.get(rank, 0))
            if owned_count >= 4 and used_count < 4:
                penalty += 18.0
            elif owned_count == 3 and used_count < 3:
                penalty += 8.0
            elif owned_count == 2 and used_count == 1:
                penalty += 4.0
        penalty += self._sequence_break_penalty(action, hand_counts)
        return penalty

    def _action_rank_counts(self, action: CardAction) -> dict[CardRank, int]:
        counts: dict[CardRank, int] = {}
        for rank in action.ranks:
            counts[rank] = counts.get(rank, 0) + 1
        return counts

    def _sequence_break_penalty(self, action: CardAction, hand_counts: Mapping[CardRank, int]) -> float:
        penalty = 0.0
        classified = self.rule_engine.classify_action(action)
        if classified.action_type not in {
            ActionType.STRAIGHT,
            ActionType.PAIR_STRAIGHT,
            ActionType.TRIPLE_STRAIGHT,
            ActionType.AIRPLANE_WITH_SINGLES,
            ActionType.AIRPLANE_WITH_PAIRS,
        }:
            action_counts = self._action_rank_counts(action)
            before_savings = self._sequence_savings(hand_counts)
            remaining_counts = dict(hand_counts)
            for rank, used_count in action_counts.items():
                remaining_counts[rank] = max(0, remaining_counts.get(rank, 0) - used_count)
            after_savings = self._sequence_savings(remaining_counts)
            if before_savings > after_savings:
                penalty = float(before_savings - after_savings) * 3.0
        return penalty

    def _primary_rank_adjustment(self, action_type: ActionType, primary_rank: CardRank) -> float:
        rank_value = float(int(primary_rank) - int(CardRank.THREE))
        adjustment = rank_value * 0.03
        if action_type in {ActionType.SINGLE, ActionType.PAIR, ActionType.TRIPLE}:
            adjustment = rank_value * -0.04
        return adjustment

    def _pass_score(self, state_view: GameStateView | None) -> float:
        score = 0.02
        if state_view is not None:
            opponent_min_count = self._minimum_opponent_hand_count(state_view)
            if opponent_min_count <= 2:
                score = 0.001
        return score

    def _endgame_pressure_bonus(self, action: CardAction, state_view: GameStateView) -> float:
        bonus = 0.0
        opponent_min_count = self._minimum_opponent_hand_count(state_view)
        if opponent_min_count <= 2 and action.card_count() >= 2:
            bonus += 3.0
        if opponent_min_count <= 1 and action.card_count() >= 1:
            bonus += 5.0
        return bonus

    def _minimum_opponent_hand_count(self, state_view: GameStateView) -> int:
        left_count = int(state_view.hand_counts.get(PlayerSeat.LEFT_OPPONENT, 17))
        right_count = int(state_view.hand_counts.get(PlayerSeat.RIGHT_OPPONENT, 17))
        minimum_count = min(left_count, right_count)
        return minimum_count

    def _bomb_bonus(self, action: CardAction, state_view: GameStateView | None) -> float:
        bonus = -18.0
        if state_view is not None:
            remaining_counts = self._remaining_counts_after_action(state_view.self_rank_counts, action)
            remaining_card_count = sum(remaining_counts.values())
            opponent_min_count = self._minimum_opponent_hand_count(state_view)
            if remaining_card_count == 0:
                bonus = 80.0
            elif opponent_min_count <= 3:
                bonus = 8.0
        return bonus

    def _rocket_bonus(self, action: CardAction, state_view: GameStateView | None) -> float:
        bonus = 2.0
        if state_view is not None:
            remaining_counts = self._remaining_counts_after_action(state_view.self_rank_counts, action)
            remaining_card_count = sum(remaining_counts.values())
            if remaining_card_count == 0:
                bonus = 90.0
        return bonus

    def _to_recommendations(
        self,
        scored_actions: tuple[tuple[CardAction, float], ...],
        reason_code: str,
        state_view: GameStateView | None = None,
    ) -> tuple[ActionRecommendation, ...]:
        total_score = sum(score for _, score in scored_actions)
        max_score = max((score for _, score in scored_actions), default=0.0)
        recommendations: list[ActionRecommendation] = []
        for action, score in scored_actions:
            probability = 0.0
            if total_score > 0.0:
                probability = score / total_score
            expected_win_rate = self._estimate_expected_win_rate(action, score, max_score)
            recommendations.append(
                ActionRecommendation(
                    action=action,
                    probability=probability,
                    expected_win_rate=expected_win_rate,
                    reason_code=reason_code,
                    reason_text=self._build_reason_text(action, state_view, reason_code),
                    risk_text=self._build_risk_text(state_view),
                    action_label=self._build_action_label(action),
                )
            )
        result = tuple(recommendations)
        return result

    def _build_reason_text(
        self,
        action: CardAction,
        state_view: GameStateView | None,
        reason_code: str,
    ) -> str:
        classified = self.rule_engine.classify_action(action)
        reasons: list[str] = []
        if classified.action_type is ActionType.PASS:
            reasons.append("不拆牌，暂避当前牌型")
        else:
            if state_view is not None and state_view.current_trick_action is not None:
                reasons.append("可压过当前牌")
            remaining_count = 0
            if state_view is not None:
                remaining_counts = self._remaining_counts_after_action(state_view.self_rank_counts, action)
                remaining_count = sum(remaining_counts.values())
                turn_estimate = self._turn_count_estimate(remaining_counts)
                reasons.append(f"剩{remaining_count}张约{turn_estimate}手")
            if classified.action_type in {ActionType.BOMB, ActionType.ROCKET}:
                if remaining_count == 0:
                    reasons.append("直接收尾")
                else:
                    reasons.append("高牌权但消耗控制牌")
            elif classified.action_type in {ActionType.STRAIGHT, ActionType.PAIR_STRAIGHT, ActionType.TRIPLE_STRAIGHT}:
                reasons.append("减少手数")
            elif classified.action_type in {ActionType.SINGLE, ActionType.PAIR}:
                reasons.append("优先小牌试探")
        if reason_code == "onnx_policy":
            reasons.append("模型评分")
        else:
            reasons.append("专家规则评分")
        text = "；".join(reasons)
        return text

    def _build_action_label(self, action: CardAction) -> str:
        classified = self.rule_engine.classify_action(action)
        ranks_text = self._format_ranks(action.ranks)
        type_name = self._action_type_name(classified.action_type)
        if classified.action_type is ActionType.PASS:
            label = "过"
        elif classified.action_type in {ActionType.STRAIGHT, ActionType.PAIR_STRAIGHT, ActionType.TRIPLE_STRAIGHT}:
            label = f"{type_name} {self._chain_label(action.ranks)}"
        else:
            label = f"{type_name} {ranks_text}"
        return label

    def _action_type_name(self, action_type: ActionType) -> str:
        names = {
            ActionType.PASS: "过",
            ActionType.SINGLE: "单张",
            ActionType.PAIR: "对子",
            ActionType.TRIPLE: "三张",
            ActionType.TRIPLE_WITH_SINGLE: "三带一",
            ActionType.TRIPLE_WITH_PAIR: "三带二",
            ActionType.STRAIGHT: "顺子",
            ActionType.PAIR_STRAIGHT: "连对",
            ActionType.TRIPLE_STRAIGHT: "飞机",
            ActionType.AIRPLANE_WITH_SINGLES: "飞机带单",
            ActionType.AIRPLANE_WITH_PAIRS: "飞机带对",
            ActionType.FOUR_WITH_TWO_SINGLES: "四带二",
            ActionType.FOUR_WITH_TWO_PAIRS: "四带两对",
            ActionType.BOMB: "炸弹",
            ActionType.ROCKET: "火箭",
            ActionType.UNKNOWN: "未知",
            ActionType.INVALID: "无效",
        }
        name = names[action_type]
        return name

    def _format_ranks(self, ranks: tuple[CardRank, ...]) -> str:
        text = ",".join(self._rank_symbol(rank) for rank in ranks)
        if text == "":
            text = "过"
        return text

    def _chain_label(self, ranks: tuple[CardRank, ...]) -> str:
        unique_ranks = tuple(sorted(set(ranks), key=int))
        if len(unique_ranks) > 1:
            label = f"{self._rank_symbol(unique_ranks[0])}-{self._rank_symbol(unique_ranks[-1])}"
        else:
            label = self._format_ranks(ranks)
        return label

    def _build_risk_text(self, state_view: GameStateView | None) -> str:
        text = ""
        if state_view is not None and self._has_reliable_public_history(state_view):
            risky_ranks: list[str] = []
            for rank, count in state_view.known_unseen_counts.items():
                if rank not in {CardRank.SMALL_JOKER, CardRank.BIG_JOKER} and count >= 4:
                    risky_ranks.append(self._rank_symbol(rank))
            has_small_joker = state_view.known_unseen_counts.get(CardRank.SMALL_JOKER, 0) > 0
            has_big_joker = state_view.known_unseen_counts.get(CardRank.BIG_JOKER, 0) > 0
            if has_small_joker and has_big_joker:
                risky_ranks.append("王炸")
            if len(risky_ranks) > 0:
                text = "风险:" + ",".join(risky_ranks)
        return text

    def _has_reliable_public_history(self, state_view: GameStateView) -> bool:
        has_history = len(state_view.action_log) > 0 or len(state_view.public_played_counts) > 0
        return has_history

    def _turn_count_estimate(self, hand_counts: Mapping[CardRank, int]) -> int:
        total_cards = sum(int(count) for count in hand_counts.values())
        if total_cards <= 12:
            turn_count = self._bounded_min_turn_count(hand_counts)
        else:
            turn_count = self._fast_turn_count_estimate(hand_counts)
        return turn_count

    def _fast_turn_count_estimate(self, hand_counts: Mapping[CardRank, int]) -> int:
        turn_count = 0
        for _, count in hand_counts.items():
            if count > 0:
                turn_count += 1
        sequence_savings = self._sequence_savings(hand_counts)
        turn_count = max(0, turn_count - sequence_savings)
        return turn_count

    def _bounded_min_turn_count(self, hand_counts: Mapping[CardRank, int]) -> int:
        key = self._hand_count_key(hand_counts)
        cached = self._turn_count_cache.get(key)
        if cached is not None:
            turn_count = cached
        else:
            total_cards = sum(int(count) for count in hand_counts.values())
            if total_cards == 0:
                turn_count = 0
            else:
                turn_count = self._fast_turn_count_estimate(hand_counts)
                actions = self.rule_engine.enumerate_legal_actions(
                    hand_counts=hand_counts,
                    actor_seat=PlayerSeat.SELF,
                    previous_action=None,
                )
                index = 0
                while index < len(actions):
                    action = actions[index]
                    if action.card_count() > 0:
                        remaining_counts = self._remaining_counts_after_action(hand_counts, action)
                        remaining_total = sum(int(count) for count in remaining_counts.values())
                        if remaining_total < total_cards:
                            candidate_turns = 1 + self._bounded_min_turn_count(remaining_counts)
                            if candidate_turns < turn_count:
                                turn_count = candidate_turns
                    index += 1
            self._turn_count_cache[key] = turn_count
        return turn_count

    def _hand_count_key(self, hand_counts: Mapping[CardRank, int]) -> tuple[int, ...]:
        values: list[int] = []
        for rank in FULL_DECK_RANK_COUNTS.keys():
            values.append(int(hand_counts.get(rank, 0)))
        key = tuple(values)
        return key

    def _sequence_savings(self, hand_counts: Mapping[CardRank, int]) -> int:
        single_savings = self._sequence_savings_for_repeat(hand_counts, 1, 5)
        pair_savings = self._sequence_savings_for_repeat(hand_counts, 2, 3)
        triple_savings = self._sequence_savings_for_repeat(hand_counts, 3, 2)
        savings = max(single_savings, pair_savings, triple_savings)
        return savings

    def _sequence_savings_for_repeat(
        self,
        hand_counts: Mapping[CardRank, int],
        repeat_count: int,
        minimum_length: int,
    ) -> int:
        best_length = 0
        current_length = 0
        rank = CardRank.THREE
        while int(rank) <= int(CardRank.ACE):
            if hand_counts.get(rank, 0) >= repeat_count:
                current_length += 1
            else:
                if current_length > best_length:
                    best_length = current_length
                current_length = 0
            rank = CardRank(int(rank) + 1) if int(rank) < int(CardRank.ACE) else CardRank.TWO
        if current_length > best_length:
            best_length = current_length
        savings = 0
        if best_length >= minimum_length:
            savings = best_length - 1
        return savings

    def _estimate_expected_win_rate(self, action: CardAction, score: float, max_score: float) -> float:
        classified = self.rule_engine.classify_action(action)
        estimated = 0.05
        if classified.action_type is not ActionType.PASS and max_score > 0.0:
            estimated = 0.2 + 0.72 * (score / max_score)
        if action.card_count() == 0:
            estimated = 0.05
        estimated = min(0.95, max(0.01, estimated))
        return estimated

    def _rank_symbol(self, rank: CardRank) -> str:
        symbols = {
            CardRank.THREE: "3",
            CardRank.FOUR: "4",
            CardRank.FIVE: "5",
            CardRank.SIX: "6",
            CardRank.SEVEN: "7",
            CardRank.EIGHT: "8",
            CardRank.NINE: "9",
            CardRank.TEN: "10",
            CardRank.JACK: "J",
            CardRank.QUEEN: "Q",
            CardRank.KING: "K",
            CardRank.ACE: "A",
            CardRank.TWO: "2",
            CardRank.SMALL_JOKER: "小王",
            CardRank.BIG_JOKER: "大王",
        }
        text = symbols[rank]
        return text

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
