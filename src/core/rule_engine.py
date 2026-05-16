from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

from src.domain.actions import ActionType, CardAction, ClassifiedAction, PlayerSeat
from src.domain.cards import CardRank, JOKER_RANKS, SEQUENCE_RANKS, count_ranks, normalize_ranks


class RuleEngine:
    def classify_action(self, action: CardAction) -> ClassifiedAction:
        ranks = normalize_ranks(action.ranks)
        rank_counts = count_ranks(ranks)
        count_values = sorted(rank_counts.values())
        unique_ranks = tuple(sorted(rank_counts.keys(), key=int))
        card_count = len(ranks)
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)

        if action.is_pass():
            result = ClassifiedAction(ActionType.PASS, None, 0, 0)

        if result.action_type is ActionType.INVALID and card_count == 1:
            result = ClassifiedAction(ActionType.SINGLE, ranks[0], card_count, 1)

        if result.action_type is ActionType.INVALID and card_count == 2:
            result = self._classify_two_cards(ranks, count_values)

        if result.action_type is ActionType.INVALID and card_count == 3:
            result = self._classify_three_cards(unique_ranks, count_values, card_count)

        if result.action_type is ActionType.INVALID and card_count == 4:
            result = self._classify_four_cards(rank_counts, count_values, card_count)

        if result.action_type is ActionType.INVALID and card_count == 5:
            result = self._classify_five_cards(rank_counts, unique_ranks, count_values, card_count)

        if result.action_type is ActionType.INVALID and card_count >= 5:
            result = self._classify_sequences(rank_counts, unique_ranks, count_values, card_count)

        if result.action_type is ActionType.INVALID and card_count >= 6:
            result = self._classify_airplane_or_four(rank_counts, unique_ranks, card_count)

        return result

    def is_legal_response(self, action: CardAction, previous_action: ClassifiedAction | None) -> bool:
        current = self.classify_action(action)
        result = False

        if current.action_type is ActionType.PASS:
            result = previous_action is not None
        elif current.action_type is not ActionType.INVALID:
            if previous_action is None:
                result = True
            elif previous_action.action_type is ActionType.PASS:
                result = True
            else:
                result = self.compare_actions(current, previous_action)

        return result

    def compare_actions(self, challenger: ClassifiedAction, incumbent: ClassifiedAction) -> bool:
        result = False

        if challenger.action_type is ActionType.ROCKET:
            result = incumbent.action_type is not ActionType.ROCKET
        elif incumbent.action_type is ActionType.ROCKET:
            result = False
        elif challenger.action_type is ActionType.BOMB and incumbent.action_type is not ActionType.BOMB:
            result = True
        elif challenger.action_type is ActionType.BOMB and incumbent.action_type is ActionType.BOMB:
            result = self._primary_greater(challenger, incumbent)
        elif challenger.action_type is incumbent.action_type:
            same_shape = (
                challenger.card_count == incumbent.card_count
                and challenger.chain_length == incumbent.chain_length
            )
            if same_shape:
                result = self._primary_greater(challenger, incumbent)

        return result

    def enumerate_legal_actions(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
        previous_action: ClassifiedAction | None,
    ) -> tuple[CardAction, ...]:
        candidates: list[CardAction] = []
        if previous_action is not None:
            candidates.append(CardAction(actor_seat=actor_seat, ranks=(), declared_type=ActionType.PASS))

        candidates.extend(self._enumerate_same_rank_actions(hand_counts, actor_seat))
        candidates.extend(self._enumerate_triple_attachments(hand_counts, actor_seat))
        candidates.extend(self._enumerate_sequences(hand_counts, actor_seat))
        candidates.extend(self._enumerate_airplanes(hand_counts, actor_seat))
        candidates.extend(self._enumerate_rocket(hand_counts, actor_seat))

        legal_actions: list[CardAction] = []
        seen: set[tuple[CardRank, ...]] = set()
        for candidate in candidates:
            legal = self.is_legal_response(candidate, previous_action)
            unseen = candidate.ranks not in seen
            if legal and unseen:
                legal_actions.append(candidate)
                seen.add(candidate.ranks)

        ordered = tuple(sorted(legal_actions, key=lambda item: (item.card_count(), tuple(int(rank) for rank in item.ranks))))
        return ordered

    def _classify_two_cards(self, ranks: tuple[CardRank, ...], count_values: list[int]) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, len(ranks), 0)
        is_pair = count_values == [2]
        is_rocket = ranks == JOKER_RANKS
        if is_rocket:
            result = ClassifiedAction(ActionType.ROCKET, CardRank.BIG_JOKER, 2, 1)
        elif is_pair:
            result = ClassifiedAction(ActionType.PAIR, ranks[0], 2, 1)
        return result

    def _classify_three_cards(
        self,
        unique_ranks: tuple[CardRank, ...],
        count_values: list[int],
        card_count: int,
    ) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)
        if count_values == [3]:
            result = ClassifiedAction(ActionType.TRIPLE, unique_ranks[0], card_count, 1)
        return result

    def _classify_four_cards(
        self,
        rank_counts: Mapping[CardRank, int],
        count_values: list[int],
        card_count: int,
    ) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)
        if count_values == [4]:
            primary_rank = self._rank_with_count(rank_counts, 4)
            result = ClassifiedAction(ActionType.BOMB, primary_rank, card_count, 1)
        elif count_values == [1, 3]:
            primary_rank = self._rank_with_count(rank_counts, 3)
            result = ClassifiedAction(ActionType.TRIPLE_WITH_SINGLE, primary_rank, card_count, 1)
        return result

    def _classify_five_cards(
        self,
        rank_counts: Mapping[CardRank, int],
        unique_ranks: tuple[CardRank, ...],
        count_values: list[int],
        card_count: int,
    ) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)
        if count_values == [2, 3]:
            primary_rank = self._rank_with_count(rank_counts, 3)
            result = ClassifiedAction(ActionType.TRIPLE_WITH_PAIR, primary_rank, card_count, 1)
        elif count_values == [1, 1, 1, 1, 1] and self._is_sequence(unique_ranks):
            result = ClassifiedAction(ActionType.STRAIGHT, unique_ranks[-1], card_count, len(unique_ranks))
        return result

    def _classify_sequences(
        self,
        rank_counts: Mapping[CardRank, int],
        unique_ranks: tuple[CardRank, ...],
        count_values: list[int],
        card_count: int,
    ) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)
        all_single = self._all_counts_equal(count_values, 1)
        all_pair = self._all_counts_equal(count_values, 2)
        all_triple = self._all_counts_equal(count_values, 3)

        if all_single and card_count >= 5 and self._is_sequence(unique_ranks):
            result = ClassifiedAction(ActionType.STRAIGHT, unique_ranks[-1], card_count, len(unique_ranks))
        elif all_pair and card_count >= 6 and card_count % 2 == 0 and self._is_sequence(unique_ranks):
            result = ClassifiedAction(ActionType.PAIR_STRAIGHT, unique_ranks[-1], card_count, len(unique_ranks))
        elif all_triple and card_count >= 6 and card_count % 3 == 0 and self._is_sequence(unique_ranks):
            result = ClassifiedAction(ActionType.TRIPLE_STRAIGHT, unique_ranks[-1], card_count, len(unique_ranks))
        else:
            result = self._classify_airplane_or_four(rank_counts, unique_ranks, card_count)

        return result

    def _classify_airplane_or_four(
        self,
        rank_counts: Mapping[CardRank, int],
        unique_ranks: tuple[CardRank, ...],
        card_count: int,
    ) -> ClassifiedAction:
        result = ClassifiedAction(ActionType.INVALID, None, card_count, 0)
        triple_ranks = tuple(sorted((rank for rank, count in rank_counts.items() if count == 3), key=int))
        four_ranks = tuple(sorted((rank for rank, count in rank_counts.items() if count == 4), key=int))
        pair_ranks = tuple(sorted((rank for rank, count in rank_counts.items() if count == 2), key=int))

        if len(four_ranks) == 1 and card_count == 6:
            result = ClassifiedAction(ActionType.FOUR_WITH_TWO_SINGLES, four_ranks[0], card_count, 1)
        elif len(four_ranks) == 1 and card_count == 8 and len(pair_ranks) == 2:
            result = ClassifiedAction(ActionType.FOUR_WITH_TWO_PAIRS, four_ranks[0], card_count, 1)
        elif len(triple_ranks) >= 2 and self._is_sequence(tuple(sorted(triple_ranks, key=int))):
            triple_count = len(triple_ranks)
            attachment_cards = card_count - triple_count * 3
            if attachment_cards == triple_count:
                result = ClassifiedAction(ActionType.AIRPLANE_WITH_SINGLES, triple_ranks[-1], card_count, triple_count)
            elif attachment_cards == triple_count * 2 and len(pair_ranks) == triple_count:
                result = ClassifiedAction(ActionType.AIRPLANE_WITH_PAIRS, triple_ranks[-1], card_count, triple_count)

        return result

    def _rank_with_count(self, rank_counts: Mapping[CardRank, int], expected_count: int) -> CardRank:
        selected = CardRank.THREE
        for rank, count in rank_counts.items():
            if count == expected_count and int(rank) >= int(selected):
                selected = rank
        return selected

    def _is_sequence(self, ranks: Sequence[CardRank]) -> bool:
        sorted_ranks = tuple(sorted(ranks, key=int))
        result = len(sorted_ranks) > 0
        index = 0
        while index < len(sorted_ranks):
            if sorted_ranks[index] not in SEQUENCE_RANKS:
                result = False
            index += 1

        index = 1
        while index < len(sorted_ranks):
            if int(sorted_ranks[index]) != int(sorted_ranks[index - 1]) + 1:
                result = False
            index += 1

        return result

    def _all_counts_equal(self, count_values: Sequence[int], expected_count: int) -> bool:
        result = len(count_values) > 0
        index = 0
        while index < len(count_values):
            if count_values[index] != expected_count:
                result = False
            index += 1
        return result

    def _primary_greater(self, challenger: ClassifiedAction, incumbent: ClassifiedAction) -> bool:
        result = False
        if challenger.primary_rank is not None and incumbent.primary_rank is not None:
            result = int(challenger.primary_rank) > int(incumbent.primary_rank)
        return result

    def _enumerate_same_rank_actions(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        for rank, count in hand_counts.items():
            if count >= 1:
                actions.append(CardAction(actor_seat=actor_seat, ranks=(rank,)))
            if count >= 2:
                actions.append(CardAction(actor_seat=actor_seat, ranks=(rank, rank)))
            if count >= 3:
                actions.append(CardAction(actor_seat=actor_seat, ranks=(rank, rank, rank)))
            if count >= 4:
                actions.append(CardAction(actor_seat=actor_seat, ranks=(rank, rank, rank, rank)))
        result = tuple(actions)
        return result

    def _enumerate_triple_attachments(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        ranks = tuple(sorted(hand_counts.keys(), key=int))
        for triple_rank in ranks:
            if hand_counts[triple_rank] >= 3:
                for single_rank in ranks:
                    if single_rank != triple_rank and hand_counts[single_rank] >= 1:
                        actions.append(
                            CardAction(actor_seat=actor_seat, ranks=(triple_rank, triple_rank, triple_rank, single_rank))
                        )
                    if single_rank != triple_rank and hand_counts[single_rank] >= 2:
                        actions.append(
                            CardAction(
                                actor_seat=actor_seat,
                                ranks=(triple_rank, triple_rank, triple_rank, single_rank, single_rank),
                            )
                        )
        result = tuple(actions)
        return result

    def _enumerate_sequences(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        sequence_ranks = tuple(rank for rank in SEQUENCE_RANKS if hand_counts.get(rank, 0) > 0)
        actions.extend(self._enumerate_fixed_sequences(sequence_ranks, hand_counts, actor_seat, 1, 5))
        pair_ranks = tuple(rank for rank in SEQUENCE_RANKS if hand_counts.get(rank, 0) >= 2)
        actions.extend(self._enumerate_fixed_sequences(pair_ranks, hand_counts, actor_seat, 2, 3))
        triple_ranks = tuple(rank for rank in SEQUENCE_RANKS if hand_counts.get(rank, 0) >= 3)
        actions.extend(self._enumerate_fixed_sequences(triple_ranks, hand_counts, actor_seat, 3, 2))
        result = tuple(actions)
        return result

    def _enumerate_fixed_sequences(
        self,
        available_ranks: Sequence[CardRank],
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
        repeat_count: int,
        min_length: int,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        start_index = 0
        while start_index < len(available_ranks):
            end_index = start_index + min_length
            while end_index <= len(available_ranks):
                chain = tuple(available_ranks[start_index:end_index])
                if self._is_sequence(chain):
                    ranks: list[CardRank] = []
                    for rank in chain:
                        if hand_counts.get(rank, 0) >= repeat_count:
                            ranks.extend((rank,) * repeat_count)
                    if len(ranks) == len(chain) * repeat_count:
                        actions.append(CardAction(actor_seat=actor_seat, ranks=tuple(ranks)))
                end_index += 1
            start_index += 1
        result = tuple(actions)
        return result

    def _enumerate_airplanes(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        triple_ranks = tuple(rank for rank in SEQUENCE_RANKS if hand_counts.get(rank, 0) >= 3)
        start_index = 0
        while start_index < len(triple_ranks):
            end_index = start_index + 2
            while end_index <= len(triple_ranks):
                chain = tuple(triple_ranks[start_index:end_index])
                if self._is_sequence(chain):
                    actions.extend(self._build_airplane_attachments(hand_counts, actor_seat, chain))
                end_index += 1
            start_index += 1
        result = tuple(actions)
        return result

    def _build_airplane_attachments(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
        chain: tuple[CardRank, ...],
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        chain_count = len(chain)
        base: list[CardRank] = []
        for rank in chain:
            base.extend((rank, rank, rank))

        single_candidates = tuple(rank for rank, count in hand_counts.items() if rank not in chain and count >= 1)
        pair_candidates = tuple(rank for rank, count in hand_counts.items() if rank not in chain and count >= 2)

        for singles in combinations(single_candidates, chain_count):
            ranks = tuple(base + list(singles))
            actions.append(CardAction(actor_seat=actor_seat, ranks=ranks))

        for pairs in combinations(pair_candidates, chain_count):
            ranks_list = list(base)
            for pair_rank in pairs:
                ranks_list.extend((pair_rank, pair_rank))
            actions.append(CardAction(actor_seat=actor_seat, ranks=tuple(ranks_list)))

        result = tuple(actions)
        return result

    def _enumerate_rocket(
        self,
        hand_counts: Mapping[CardRank, int],
        actor_seat: PlayerSeat,
    ) -> tuple[CardAction, ...]:
        actions: list[CardAction] = []
        has_small = hand_counts.get(CardRank.SMALL_JOKER, 0) >= 1
        has_big = hand_counts.get(CardRank.BIG_JOKER, 0) >= 1
        if has_small and has_big:
            actions.append(CardAction(actor_seat=actor_seat, ranks=JOKER_RANKS))
        result = tuple(actions)
        return result
