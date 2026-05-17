from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.core.rule_engine import RuleEngine
from src.domain.actions import ActionLogEntry, ActionType, CardAction, GameStateView, PlayerSeat
from src.domain.cards import CardRank, FULL_DECK_RANK_COUNTS


PERFECTDOU_INPUT_SIZE = 32738
PERFECTDOU_OUTPUT_SIZE = 621
PERFECTDOU_STATE_SIZE = 4680
PERFECTDOU_LSTM_STATE_SIZE = 8
PERFECTDOU_ACTION_OFFSET = PERFECTDOU_STATE_SIZE + PERFECTDOU_LSTM_STATE_SIZE
PERFECTDOU_MAX_ACTIONS = 150
PERFECTDOU_ACTION_STRIDE = 187
PERFECTDOU_ACTION_ID_COLUMN = 0
PERFECTDOU_ACTION_VALID_COLUMN = 6


POLICY_RANKS: tuple[CardRank, ...] = (
    CardRank.THREE,
    CardRank.FOUR,
    CardRank.FIVE,
    CardRank.SIX,
    CardRank.SEVEN,
    CardRank.EIGHT,
    CardRank.NINE,
    CardRank.TEN,
    CardRank.JACK,
    CardRank.QUEEN,
    CardRank.KING,
    CardRank.ACE,
    CardRank.TWO,
)

POLICY_SEQUENCE_RANKS: tuple[CardRank, ...] = (
    CardRank.THREE,
    CardRank.FOUR,
    CardRank.FIVE,
    CardRank.SIX,
    CardRank.SEVEN,
    CardRank.EIGHT,
    CardRank.NINE,
    CardRank.TEN,
    CardRank.JACK,
    CardRank.QUEEN,
    CardRank.KING,
    CardRank.ACE,
)

POLICY_SINGLE_RANKS: tuple[CardRank, ...] = (
    *POLICY_RANKS,
    CardRank.SMALL_JOKER,
    CardRank.BIG_JOKER,
)


@dataclass(frozen=True, slots=True)
class PerfectDouActionRef:
    action: CardAction
    action_index: int
    action_key: str


class PerfectDouAdapter:
    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine
        self.action_space = build_perfectdou_action_space()

    def action_key_for_action(self, action: CardAction) -> str | None:
        classified = self.rule_engine.classify_action(action)
        key: str | None = None
        if classified.action_type is ActionType.PASS:
            key = "pass"
        elif classified.action_type is ActionType.ROCKET:
            key = "BR"
        elif classified.action_type in {
            ActionType.AIRPLANE_WITH_SINGLES,
            ActionType.AIRPLANE_WITH_PAIRS,
            ActionType.FOUR_WITH_TWO_SINGLES,
            ActionType.FOUR_WITH_TWO_PAIRS,
        }:
            key = self._abstract_action_key(action, classified.action_type)
        elif classified.action_type is not ActionType.INVALID:
            key = "".join(_rank_char(rank) for rank in action.ranks)
        return key

    def action_index_for_action(self, action: CardAction) -> int | None:
        key = self.action_key_for_action(action)
        index: int | None = None
        if key is not None:
            value = self.action_space.get(key)
            if value is not None:
                index = int(value)
        return index

    def encode_policy_input(
        self,
        state_view: GameStateView,
        legal_actions: Sequence[CardAction],
    ) -> np.ndarray:
        tensor = np.zeros((1, PERFECTDOU_INPUT_SIZE), dtype=np.float32)
        state_planes = self._encode_state_planes(state_view)
        tensor[0, 0:PERFECTDOU_STATE_SIZE] = state_planes.reshape(-1)
        self._encode_action_rows(tensor, state_view, legal_actions)
        return tensor

    def _abstract_action_key(self, action: CardAction, action_type: ActionType) -> str | None:
        counts = Counter(action.ranks)
        key: str | None = None
        if action_type in {ActionType.AIRPLANE_WITH_SINGLES, ActionType.AIRPLANE_WITH_PAIRS}:
            triple_ranks = tuple(sorted((rank for rank, count in counts.items() if count >= 3), key=int))
            if len(triple_ranks) > 0:
                base = "".join(_rank_char(rank) * 3 for rank in triple_ranks)
                star_count = len(triple_ranks)
                if action_type is ActionType.AIRPLANE_WITH_PAIRS:
                    star_count = len(triple_ranks) * 2
                key = base + ("*" * star_count)
        elif action_type in {ActionType.FOUR_WITH_TWO_SINGLES, ActionType.FOUR_WITH_TWO_PAIRS}:
            four_rank = self._rank_with_count(counts, 4)
            if four_rank is not None:
                star_count = 2
                if action_type is ActionType.FOUR_WITH_TWO_PAIRS:
                    star_count = 4
                key = (_rank_char(four_rank) * 4) + ("*" * star_count)
        return key

    def _rank_with_count(self, counts: Mapping[CardRank, int], expected_count: int) -> CardRank | None:
        selected: CardRank | None = None
        for rank, count in counts.items():
            if count == expected_count and (selected is None or int(rank) > int(selected)):
                selected = rank
        return selected

    def _encode_state_planes(self, state_view: GameStateView) -> np.ndarray:
        planes = np.zeros((26, 12, 15), dtype=np.float32)
        self._write_count_plane(planes, 2, state_view.self_rank_counts)
        self._write_count_plane(planes, 3, state_view.known_unseen_counts)
        self._write_count_plane(planes, 19, state_view.public_played_counts)
        if state_view.current_trick_action is not None and state_view.current_trick_action.primary_rank is not None:
            trick_counts = {state_view.current_trick_action.primary_rank: state_view.current_trick_action.card_count}
            self._write_count_plane(planes, 20, trick_counts)
        self._write_count_plane(planes, 21, state_view.self_rank_counts)
        self._write_count_plane(planes, 22, state_view.known_unseen_counts)
        self._write_hand_count_plane(planes, 23, state_view.hand_counts)
        self._write_turn_plane(planes, 24, state_view.current_turn)
        self._write_turn_plane(planes, 25, state_view.landlord_seat)
        self._write_recent_actions(planes, state_view.action_log)
        return planes

    def _write_recent_actions(self, planes: np.ndarray, action_log: tuple[ActionLogEntry, ...]) -> None:
        recent_actions = action_log[-15:]
        start_plane = 4 + max(0, 15 - len(recent_actions))
        index = 0
        while index < len(recent_actions):
            self._write_rank_tuple_plane(planes, start_plane + index, recent_actions[index].action.ranks)
            index += 1

    def _write_count_plane(
        self,
        planes: np.ndarray,
        plane_index: int,
        counts: Mapping[CardRank, int],
    ) -> None:
        for rank in POLICY_RANKS:
            rank_index = _rank_index(rank)
            amount = min(4, max(0, int(counts.get(rank, 0))))
            row = 0
            while row < amount:
                planes[plane_index, row, rank_index] = 1.0
                row += 1
        small_joker = 1.0 if int(counts.get(CardRank.SMALL_JOKER, 0)) > 0 else 0.0
        big_joker = 1.0 if int(counts.get(CardRank.BIG_JOKER, 0)) > 0 else 0.0
        planes[plane_index, 0, _rank_index(CardRank.SMALL_JOKER)] = small_joker
        planes[plane_index, 0, _rank_index(CardRank.BIG_JOKER)] = big_joker

    def _write_rank_tuple_plane(self, planes: np.ndarray, plane_index: int, ranks: tuple[CardRank, ...]) -> None:
        self._write_count_plane(planes, plane_index, Counter(ranks))

    def _write_hand_count_plane(
        self,
        planes: np.ndarray,
        plane_index: int,
        hand_counts: Mapping[PlayerSeat, int],
    ) -> None:
        seats = (PlayerSeat.SELF, PlayerSeat.LEFT_OPPONENT, PlayerSeat.RIGHT_OPPONENT)
        seat_index = 0
        while seat_index < len(seats):
            amount = max(0, min(20, int(hand_counts.get(seats[seat_index], 0))))
            column = min(14, amount * 14 // 20)
            planes[plane_index, seat_index, column] = 1.0
            seat_index += 1

    def _write_turn_plane(self, planes: np.ndarray, plane_index: int, seat: PlayerSeat) -> None:
        seat_columns = {
            PlayerSeat.SELF: 0,
            PlayerSeat.LEFT_OPPONENT: 1,
            PlayerSeat.RIGHT_OPPONENT: 2,
            PlayerSeat.TABLE: 3,
        }
        planes[plane_index, 0, seat_columns.get(seat, 0)] = 1.0

    def _encode_action_rows(
        self,
        tensor: np.ndarray,
        state_view: GameStateView,
        legal_actions: Sequence[CardAction],
    ) -> None:
        row_index = 0
        while row_index < len(legal_actions) and row_index < PERFECTDOU_MAX_ACTIONS:
            action = legal_actions[row_index]
            action_index = self.action_index_for_action(action)
            if action_index is not None:
                row_start = PERFECTDOU_ACTION_OFFSET + row_index * PERFECTDOU_ACTION_STRIDE
                row = tensor[0, row_start : row_start + PERFECTDOU_ACTION_STRIDE]
                self._encode_action_row(row, state_view, action, action_index)
            row_index += 1

    def _encode_action_row(
        self,
        row: np.ndarray,
        state_view: GameStateView,
        action: CardAction,
        action_index: int,
    ) -> None:
        classified = self.rule_engine.classify_action(action)
        row[PERFECTDOU_ACTION_ID_COLUMN] = float(action_index)
        row[PERFECTDOU_ACTION_VALID_COLUMN] = 1.0
        row[1] = float(_action_type_index(classified.action_type)) / 14.0
        row[2] = float(_rank_index(classified.primary_rank)) / 14.0 if classified.primary_rank is not None else 0.0
        row[3] = float(min(20, action.card_count())) / 20.0
        row[4] = float(min(12, classified.chain_length)) / 12.0
        row[5] = 1.0 if classified.action_type in {ActionType.BOMB, ActionType.ROCKET} else 0.0
        row[7] = 1.0 if classified.action_type is ActionType.PASS else 0.0
        self._write_flat_cards(row, 8, Counter(action.ranks))
        remaining_counts = self._remaining_counts_after_action(state_view.self_rank_counts, action)
        self._write_flat_cards(row, 62, remaining_counts)
        self._write_rank_counts(row, 116, Counter(action.ranks))
        self._write_rank_counts(row, 131, state_view.self_rank_counts)
        self._write_rank_counts(row, 146, remaining_counts)
        row[161] = 1.0 if sum(remaining_counts.values()) == 0 else 0.0
        row[162] = 1.0 if state_view.current_trick_action is not None else 0.0
        row[163] = float(min(20, sum(remaining_counts.values()))) / 20.0
        row[164] = float(min(20, int(state_view.hand_counts.get(PlayerSeat.LEFT_OPPONENT, 17)))) / 20.0
        row[165] = float(min(20, int(state_view.hand_counts.get(PlayerSeat.RIGHT_OPPONENT, 17)))) / 20.0
        row[166] = 1.0 if state_view.landlord_seat is PlayerSeat.SELF else 0.0

    def _remaining_counts_after_action(
        self,
        hand_counts: Mapping[CardRank, int],
        action: CardAction,
    ) -> dict[CardRank, int]:
        remaining = {rank: int(count) for rank, count in hand_counts.items() if int(count) > 0}
        for rank in action.ranks:
            next_count = max(0, remaining.get(rank, 0) - 1)
            if next_count > 0:
                remaining[rank] = next_count
            else:
                remaining.pop(rank, None)
        return remaining

    def _write_flat_cards(self, row: np.ndarray, offset: int, counts: Mapping[CardRank, int]) -> None:
        cursor = offset
        for rank in POLICY_RANKS:
            amount = min(4, max(0, int(counts.get(rank, 0))))
            slot = 0
            while slot < 4:
                row[cursor] = 1.0 if slot < amount else 0.0
                cursor += 1
                slot += 1
        row[cursor] = 1.0 if int(counts.get(CardRank.SMALL_JOKER, 0)) > 0 else 0.0
        row[cursor + 1] = 1.0 if int(counts.get(CardRank.BIG_JOKER, 0)) > 0 else 0.0

    def _write_rank_counts(self, row: np.ndarray, offset: int, counts: Mapping[CardRank, int]) -> None:
        ranks = POLICY_SINGLE_RANKS
        index = 0
        while index < len(ranks):
            rank = ranks[index]
            full_count = max(1, FULL_DECK_RANK_COUNTS[rank])
            row[offset + index] = float(max(0, int(counts.get(rank, 0)))) / float(full_count)
            index += 1


def build_perfectdou_action_space() -> dict[str, int]:
    action_space: dict[str, int] = {}
    index = 0
    index = _append_rank_repeats(action_space, index, POLICY_SINGLE_RANKS, 1)
    index = _append_rank_repeats(action_space, index, POLICY_RANKS, 2)
    index = _append_rank_repeats(action_space, index, POLICY_RANKS, 3)
    index = _append_triple_with_single(action_space, index)
    index = _append_triple_with_pair(action_space, index)
    index = _append_sequences(action_space, index, repeat_count=1, minimum_length=5, maximum_length=12)
    index = _append_sequences(action_space, index, repeat_count=2, minimum_length=3, maximum_length=10)
    index = _append_sequences(action_space, index, repeat_count=3, minimum_length=2, maximum_length=6)
    index = _append_abstract_airplanes(action_space, index, pair_attachments=False)
    index = _append_abstract_airplanes(action_space, index, pair_attachments=True)
    index = _append_four_attachments(action_space, index, star_count=2)
    index = _append_four_attachments(action_space, index, star_count=4)
    index = _append_rank_repeats(action_space, index, POLICY_RANKS, 4)
    action_space["BR"] = index
    index += 1
    action_space["pass"] = index
    return action_space


def _append_rank_repeats(
    action_space: dict[str, int],
    index: int,
    ranks: tuple[CardRank, ...],
    repeat_count: int,
) -> int:
    next_index = index
    for rank in ranks:
        action_space[_rank_char(rank) * repeat_count] = next_index
        next_index += 1
    return next_index


def _append_triple_with_single(action_space: dict[str, int], index: int) -> int:
    next_index = index
    for triple_rank in POLICY_RANKS:
        for kicker_rank in POLICY_SINGLE_RANKS:
            if kicker_rank is not triple_rank:
                key = "".join(sorted((_rank_char(triple_rank) * 3) + _rank_char(kicker_rank), key=_rank_sort_value))
                action_space[key] = next_index
                next_index += 1
    return next_index


def _append_triple_with_pair(action_space: dict[str, int], index: int) -> int:
    next_index = index
    for triple_rank in POLICY_RANKS:
        for pair_rank in POLICY_RANKS:
            if pair_rank is not triple_rank:
                key = "".join(sorted((_rank_char(triple_rank) * 3) + (_rank_char(pair_rank) * 2), key=_rank_sort_value))
                action_space[key] = next_index
                next_index += 1
    return next_index


def _append_sequences(
    action_space: dict[str, int],
    index: int,
    repeat_count: int,
    minimum_length: int,
    maximum_length: int,
) -> int:
    next_index = index
    length = minimum_length
    while length <= maximum_length:
        start = 0
        while start + length <= len(POLICY_SEQUENCE_RANKS):
            chain = POLICY_SEQUENCE_RANKS[start : start + length]
            key = "".join(_rank_char(rank) * repeat_count for rank in chain)
            action_space[key] = next_index
            next_index += 1
            start += 1
        length += 1
    return next_index


def _append_abstract_airplanes(action_space: dict[str, int], index: int, pair_attachments: bool) -> int:
    next_index = index
    maximum_length = 4 if pair_attachments else 5
    length = 2
    while length <= maximum_length:
        start = 0
        while start + length <= len(POLICY_SEQUENCE_RANKS):
            chain = POLICY_SEQUENCE_RANKS[start : start + length]
            star_count = length * 2 if pair_attachments else length
            key = "".join(_rank_char(rank) * 3 for rank in chain) + ("*" * star_count)
            action_space[key] = next_index
            next_index += 1
            start += 1
        length += 1
    return next_index


def _append_four_attachments(action_space: dict[str, int], index: int, star_count: int) -> int:
    next_index = index
    for rank in POLICY_RANKS:
        action_space[(_rank_char(rank) * 4) + ("*" * star_count)] = next_index
        next_index += 1
    return next_index


def _rank_char(rank: CardRank) -> str:
    chars = {
        CardRank.THREE: "3",
        CardRank.FOUR: "4",
        CardRank.FIVE: "5",
        CardRank.SIX: "6",
        CardRank.SEVEN: "7",
        CardRank.EIGHT: "8",
        CardRank.NINE: "9",
        CardRank.TEN: "T",
        CardRank.JACK: "J",
        CardRank.QUEEN: "Q",
        CardRank.KING: "K",
        CardRank.ACE: "A",
        CardRank.TWO: "2",
        CardRank.SMALL_JOKER: "B",
        CardRank.BIG_JOKER: "R",
    }
    return chars[rank]


def _rank_sort_value(value: str) -> int:
    order = {
        "3": 0,
        "4": 1,
        "5": 2,
        "6": 3,
        "7": 4,
        "8": 5,
        "9": 6,
        "T": 7,
        "J": 8,
        "Q": 9,
        "K": 10,
        "A": 11,
        "2": 12,
        "B": 13,
        "R": 14,
        "*": 15,
    }
    return order[value]


def _rank_index(rank: CardRank | None) -> int:
    index = 0
    if rank is not None:
        index = int(_rank_sort_value(_rank_char(rank)))
    return index


def _action_type_index(action_type: ActionType) -> int:
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
    selected = 0
    index = 0
    while index < len(ordered_types):
        if ordered_types[index] is action_type:
            selected = index
        index += 1
    return selected
