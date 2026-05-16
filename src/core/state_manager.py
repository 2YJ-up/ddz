from __future__ import annotations

from collections import Counter
from typing import Mapping

from src.core.rule_engine import RuleEngine
from src.domain.actions import (
    ActionLogEntry,
    CardAction,
    ClassifiedAction,
    GameStateView,
    InitialDeal,
    NEXT_SEAT,
    PlayerSeat,
)
from src.domain.cards import CardRank, FULL_DECK_RANK_COUNTS


class StateManager:
    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine
        self._has_initial_deal = False
        self._initial_deal = InitialDeal(
            self_cards=(),
            landlord_cards=(),
            landlord_seat=PlayerSeat.SELF,
            first_turn=PlayerSeat.SELF,
        )
        self._current_turn = PlayerSeat.SELF
        self._action_log: list[ActionLogEntry] = []

    def start_hand(self, initial_deal: InitialDeal) -> None:
        self._validate_initial_deal(initial_deal)
        self._initial_deal = initial_deal
        self._current_turn = initial_deal.first_turn
        self._action_log = []
        self._has_initial_deal = True

    def is_started(self) -> bool:
        result = self._has_initial_deal
        return result

    def append_action(self, action: CardAction) -> None:
        self._require_started()
        previous_action = self.derive_current_trick_action()
        if action.actor_seat is not self._current_turn:
            raise ValueError("action actor does not match current turn")
        if not self.rule_engine.is_legal_response(action, previous_action):
            raise ValueError("action is not legal in the current trick")
        entry = ActionLogEntry(action=action, sequence_index=len(self._action_log))
        self._action_log.append(entry)
        self._current_turn = NEXT_SEAT[action.actor_seat]

    def get_current_turn(self) -> PlayerSeat:
        self._require_started()
        current_turn = self._current_turn
        return current_turn

    def get_initial_deal(self) -> InitialDeal:
        self._require_started()
        initial_deal = self._initial_deal
        return initial_deal

    def get_action_log(self) -> tuple[ActionLogEntry, ...]:
        self._require_started()
        action_log = tuple(self._action_log)
        return action_log

    def derive_current_trick_action(self) -> ClassifiedAction | None:
        self._require_started()
        leading_action: ClassifiedAction | None = None
        pass_count = 0
        for entry in self._action_log:
            classified = self.rule_engine.classify_action(entry.action)
            if classified.action_type.value == "pass":
                if leading_action is not None:
                    pass_count += 1
                    if pass_count >= 2:
                        leading_action = None
                        pass_count = 0
            else:
                leading_action = classified
                pass_count = 0
        return leading_action

    def derive_public_played_counts(self) -> dict[CardRank, int]:
        self._require_started()
        counts: Counter[CardRank] = Counter()
        for entry in self._action_log:
            counts.update(entry.action.ranks)
        result = dict(counts)
        return result

    def derive_self_rank_counts(self) -> dict[CardRank, int]:
        self._require_started()
        counts: Counter[CardRank] = Counter(self._initial_deal.self_cards)
        if self._initial_deal.landlord_seat is PlayerSeat.SELF:
            counts.update(self._initial_deal.landlord_cards)
        for entry in self._action_log:
            if entry.action.actor_seat is PlayerSeat.SELF:
                self._subtract_counter(counts, entry.action.ranks)
        result = dict(counts)
        return result

    def derive_known_unseen_counts(self) -> dict[CardRank, int]:
        self._require_started()
        counts = dict(FULL_DECK_RANK_COUNTS)
        for entry in self._action_log:
            self._subtract_ranks(counts, entry.action.ranks)
        result = counts
        return result

    def derive_hand_counts(self) -> dict[PlayerSeat, int]:
        self._require_started()
        counts = {
            PlayerSeat.SELF: len(self._initial_deal.self_cards),
            PlayerSeat.LEFT_OPPONENT: 17,
            PlayerSeat.RIGHT_OPPONENT: 17,
        }
        if self._initial_deal.landlord_seat is PlayerSeat.LEFT_OPPONENT:
            counts[PlayerSeat.LEFT_OPPONENT] = 20
        elif self._initial_deal.landlord_seat is PlayerSeat.RIGHT_OPPONENT:
            counts[PlayerSeat.RIGHT_OPPONENT] = 20
        elif self._initial_deal.landlord_seat is PlayerSeat.SELF:
            counts[PlayerSeat.SELF] = len(self._initial_deal.self_cards) + len(self._initial_deal.landlord_cards)

        for entry in self._action_log:
            counts[entry.action.actor_seat] = counts[entry.action.actor_seat] - entry.action.card_count()

        return counts

    def build_state_matrix(self) -> tuple[int, ...]:
        self._require_started()
        unseen_counts = self.derive_known_unseen_counts()
        matrix: list[int] = []
        for rank, full_count in FULL_DECK_RANK_COUNTS.items():
            unseen_count = unseen_counts.get(rank, 0)
            index = 0
            while index < full_count:
                value = 1 if index < unseen_count else 0
                matrix.append(value)
                index += 1
        result = tuple(matrix)
        return result

    def build_view(self) -> GameStateView:
        self._require_started()
        view = GameStateView(
            current_turn=self._current_turn,
            landlord_seat=self._initial_deal.landlord_seat,
            action_log=tuple(self._action_log),
            current_trick_action=self.derive_current_trick_action(),
            self_rank_counts=self.derive_self_rank_counts(),
            public_played_counts=self.derive_public_played_counts(),
            known_unseen_counts=self.derive_known_unseen_counts(),
            hand_counts=self.derive_hand_counts(),
            state_matrix=self.build_state_matrix(),
        )
        return view

    def reset(self) -> None:
        self._has_initial_deal = False
        self._initial_deal = InitialDeal(
            self_cards=(),
            landlord_cards=(),
            landlord_seat=PlayerSeat.SELF,
            first_turn=PlayerSeat.SELF,
        )
        self._current_turn = PlayerSeat.SELF
        self._action_log = []

    def _validate_initial_deal(self, initial_deal: InitialDeal) -> None:
        combined_counts: Counter[CardRank] = Counter()
        combined_counts.update(initial_deal.self_cards)
        combined_counts.update(initial_deal.landlord_cards)
        for rank, count in combined_counts.items():
            allowed = FULL_DECK_RANK_COUNTS[rank]
            if count > allowed:
                raise ValueError("initial deal exceeds full deck rank count")

    def _require_started(self) -> None:
        if not self._has_initial_deal:
            raise RuntimeError("hand has not been started")

    def _subtract_ranks(self, counts: dict[CardRank, int], ranks: tuple[CardRank, ...]) -> None:
        local_counts: Counter[CardRank] = Counter(ranks)
        for rank, amount in local_counts.items():
            next_value = counts[rank] - amount
            if next_value < 0:
                raise ValueError("rank count became negative during state derivation")
            counts[rank] = next_value

    def _subtract_counter(self, counts: Counter[CardRank], ranks: tuple[CardRank, ...]) -> None:
        local_counts: Counter[CardRank] = Counter(ranks)
        for rank, amount in local_counts.items():
            next_value = counts[rank] - amount
            if next_value < 0:
                raise ValueError("self rank count became negative during state derivation")
            counts[rank] = next_value

    def export_base_truths(self) -> Mapping[str, object]:
        self._require_started()
        exported = {
            "initial_deal": self._initial_deal,
            "current_turn": self._current_turn,
            "action_log": tuple(self._action_log),
        }
        return exported
