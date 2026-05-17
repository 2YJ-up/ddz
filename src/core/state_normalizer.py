from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from src.core.state_manager import StateManager
from src.domain.actions import CardAction, CardDetection, GameStateView, InitialDeal, PlayerSeat
from src.domain.cards import CardRank, FULL_DECK_RANK_COUNTS


@dataclass(frozen=True, slots=True)
class StateNormalizerConfig:
    landlord_seat: PlayerSeat = PlayerSeat.SELF
    first_turn: PlayerSeat = PlayerSeat.SELF
    bootstrap_from_self_detections: bool = True
    min_self_cards_to_start: int = 1
    bootstrap_stable_frames: int = 1
    live_self_stable_frames: int = 2


class StateObservationNormalizer:
    def __init__(self, config: StateNormalizerConfig, state_manager: StateManager) -> None:
        self.config = config
        self.state_manager = state_manager
        self._candidate_self_cards: tuple[CardRank, ...] = ()
        self._candidate_stable_frames = 0
        self._accepted_live_self_cards: tuple[CardRank, ...] = ()
        self._live_candidate_self_cards: tuple[CardRank, ...] = ()
        self._live_candidate_stable_frames = 0

    def normalize(self, detections: tuple[CardDetection, ...]) -> GameStateView:
        observed_self_cards = self._extract_self_cards(detections)
        observed_table_cards = self._extract_table_cards(detections)
        self_cards = self._stabilize_self_cards(observed_self_cards)
        if not self.state_manager.is_started():
            if len(self_cards) >= self.config.min_self_cards_to_start:
                self._start_observed_hand(self_cards)
                view = self.state_manager.build_view()
            else:
                view = self.state_manager.build_waiting_view()
        else:
            self_cards = self._stabilize_live_self_cards(observed_self_cards)
            if self._should_rebootstrap(self_cards):
                self._start_observed_hand(self_cards)
            elif len(self_cards) > 0:
                self._append_self_action_from_observation(self_cards, detections)
            view = self.state_manager.build_view()
        view = self._apply_live_self_observation(view, self_cards)
        view = self._apply_table_observation(view, observed_table_cards)
        return view

    def start_hand(self, initial_deal: InitialDeal) -> None:
        self.state_manager.start_hand(initial_deal)

    def append_action(self, action: CardAction) -> None:
        self.state_manager.append_action(action)

    def _start_observed_hand(self, observed_self_cards: tuple[CardRank, ...]) -> None:
        self_cards = ()
        if self.config.bootstrap_from_self_detections:
            self_cards = observed_self_cards
        self._accept_live_self_cards(self_cards)
        initial_deal = InitialDeal(
            self_cards=self_cards,
            landlord_cards=(),
            landlord_seat=self.config.landlord_seat,
            first_turn=self.config.first_turn,
        )
        self.state_manager.start_hand(initial_deal)

    def _should_rebootstrap(self, self_cards: tuple[CardRank, ...]) -> bool:
        should_rebootstrap = False
        if len(self_cards) >= self.config.min_self_cards_to_start:
            initial_deal = self.state_manager.get_initial_deal()
            action_log = self.state_manager.get_action_log()
            current_self_cards = initial_deal.self_cards
            should_rebootstrap = len(action_log) == 0 and len(self_cards) > len(current_self_cards)
        return should_rebootstrap

    def _append_self_action_from_observation(
        self,
        observed_self_cards: tuple[CardRank, ...],
        detections: tuple[CardDetection, ...],
    ) -> None:
        view = self.state_manager.build_view()
        if view.current_turn is PlayerSeat.SELF:
            expected_counts = dict(view.self_rank_counts)
            observed_counts = self._count_ranks(observed_self_cards)
            played_ranks = self._derive_removed_ranks(expected_counts, observed_counts)
            if len(played_ranks) > 0:
                timestamp_ms = self._latest_detection_timestamp(detections)
                action = CardAction(
                    actor_seat=PlayerSeat.SELF,
                    ranks=played_ranks,
                    observed_at_ms=timestamp_ms,
                    source_frame_id="cv_observation",
                )
                try:
                    self.state_manager.append_action(action)
                except ValueError:
                    self._ignore_invalid_observed_action()

    def _derive_removed_ranks(
        self,
        expected_counts: dict[CardRank, int],
        observed_counts: dict[CardRank, int],
    ) -> tuple[CardRank, ...]:
        removed: list[CardRank] = []
        for rank, expected_count in expected_counts.items():
            observed_count = observed_counts.get(rank, 0)
            missing_count = expected_count - observed_count
            index = 0
            while index < missing_count:
                removed.append(rank)
                index += 1
        result = tuple(sorted(removed, key=int))
        return result

    def _count_ranks(self, ranks: tuple[CardRank, ...]) -> dict[CardRank, int]:
        counts: dict[CardRank, int] = {}
        for rank in ranks:
            counts[rank] = counts.get(rank, 0) + 1
        return counts

    def _latest_detection_timestamp(self, detections: tuple[CardDetection, ...]) -> int:
        timestamp_ms = 0
        for detection in detections:
            timestamp_ms = max(timestamp_ms, int(detection.confidence * 1000.0))
        return timestamp_ms

    def _ignore_invalid_observed_action(self) -> None:
        return None

    def _extract_self_cards(self, detections: tuple[CardDetection, ...]) -> tuple[CardRank, ...]:
        result = self._extract_cards_for_region(detections, PlayerSeat.SELF)
        return result

    def _extract_table_cards(self, detections: tuple[CardDetection, ...]) -> tuple[CardRank, ...]:
        result = self._extract_cards_for_region(detections, PlayerSeat.TABLE)
        return result

    def _extract_cards_for_region(
        self,
        detections: tuple[CardDetection, ...],
        seat_region: PlayerSeat,
    ) -> tuple[CardRank, ...]:
        ordered = tuple(sorted(detections, key=lambda item: (item.bbox.x, item.bbox.y, int(item.card_rank))))
        ranks = []
        rank_counts: dict[CardRank, int] = {}
        for detection in ordered:
            if detection.seat_region is seat_region:
                current_count = rank_counts.get(detection.card_rank, 0)
                allowed_count = FULL_DECK_RANK_COUNTS[detection.card_rank]
                if current_count < allowed_count:
                    ranks.append(detection.card_rank)
                    rank_counts[detection.card_rank] = current_count + 1
        result = tuple(ranks)
        return result

    def _stabilize_self_cards(self, observed_self_cards: tuple[CardRank, ...]) -> tuple[CardRank, ...]:
        required_frames = max(1, self.config.bootstrap_stable_frames)
        if observed_self_cards == self._candidate_self_cards:
            self._candidate_stable_frames += 1
        else:
            self._candidate_self_cards = observed_self_cards
            self._candidate_stable_frames = 1

        stable_cards: tuple[CardRank, ...] = ()
        if self._candidate_stable_frames >= required_frames:
            stable_cards = self._candidate_self_cards
        return stable_cards

    def _stabilize_live_self_cards(self, observed_self_cards: tuple[CardRank, ...]) -> tuple[CardRank, ...]:
        if len(self._accepted_live_self_cards) == 0:
            self._accept_live_self_cards(observed_self_cards)
        elif observed_self_cards == self._accepted_live_self_cards:
            self._reset_live_candidate()
        elif len(observed_self_cards) > len(self._accepted_live_self_cards):
            self._accept_live_self_cards(observed_self_cards)
        elif self._live_candidate_is_stable(observed_self_cards):
            self._accept_live_self_cards(observed_self_cards)
        stable_cards = self._accepted_live_self_cards
        return stable_cards

    def _live_candidate_is_stable(self, observed_self_cards: tuple[CardRank, ...]) -> bool:
        if observed_self_cards == self._live_candidate_self_cards:
            self._live_candidate_stable_frames += 1
        else:
            self._live_candidate_self_cards = observed_self_cards
            self._live_candidate_stable_frames = 1
        required_frames = max(1, self.config.live_self_stable_frames)
        candidate_is_empty_drop = len(observed_self_cards) == 0 and len(self._accepted_live_self_cards) > 3
        stable = self._live_candidate_stable_frames >= required_frames and not candidate_is_empty_drop
        return stable

    def _accept_live_self_cards(self, self_cards: tuple[CardRank, ...]) -> None:
        self._accepted_live_self_cards = self_cards
        self._reset_live_candidate()

    def _reset_live_candidate(self) -> None:
        self._live_candidate_self_cards = ()
        self._live_candidate_stable_frames = 0

    def _apply_table_observation(
        self,
        view: GameStateView,
        observed_table_cards: tuple[CardRank, ...],
    ) -> GameStateView:
        adjusted_view = view
        if len(observed_table_cards) > 0:
            table_action = CardAction(
                actor_seat=PlayerSeat.RIGHT_OPPONENT,
                ranks=observed_table_cards,
                observed_at_ms=0,
                source_frame_id="cv_table_observation",
            )
            classified = self.state_manager.rule_engine.classify_action(table_action)
            if classified.is_valid() and not table_action.is_pass():
                public_played_counts = self._merge_counts(view.public_played_counts, self._count_ranks(observed_table_cards))
                known_unseen_counts = self._derive_live_known_unseen_counts(view.self_rank_counts, public_played_counts)
                adjusted_view = replace(
                    view,
                    current_turn=PlayerSeat.SELF,
                    current_trick_action=classified,
                    public_played_counts=public_played_counts,
                    known_unseen_counts=known_unseen_counts,
                    state_matrix=self._build_state_matrix_from_known_unseen_counts(known_unseen_counts),
                )
        return adjusted_view

    def _apply_live_self_observation(
        self,
        view: GameStateView,
        observed_self_cards: tuple[CardRank, ...],
    ) -> GameStateView:
        adjusted_view = view
        if len(observed_self_cards) > 0:
            self_counts = self._count_ranks(observed_self_cards)
            hand_counts = dict(view.hand_counts)
            hand_counts[PlayerSeat.SELF] = len(observed_self_cards)
            known_unseen_counts = self._derive_live_known_unseen_counts(self_counts, view.public_played_counts)
            adjusted_view = replace(
                view,
                self_rank_counts=self_counts,
                hand_counts=hand_counts,
                known_unseen_counts=known_unseen_counts,
                state_matrix=self._build_state_matrix_from_known_unseen_counts(known_unseen_counts),
            )
        return adjusted_view

    def _derive_live_known_unseen_counts(
        self,
        self_counts: Mapping[CardRank, int],
        public_played_counts: Mapping[CardRank, int],
    ) -> dict[CardRank, int]:
        known_unseen_counts = dict(FULL_DECK_RANK_COUNTS)
        self._subtract_count_mapping_safely(known_unseen_counts, dict(self_counts))
        self._subtract_count_mapping_safely(known_unseen_counts, dict(public_played_counts))
        return known_unseen_counts

    def _merge_counts(
        self,
        base_counts: Mapping[CardRank, int],
        extra_counts: dict[CardRank, int],
    ) -> dict[CardRank, int]:
        merged = dict(base_counts)
        for rank, amount in extra_counts.items():
            merged[rank] = merged.get(rank, 0) + amount
        return merged

    def _subtract_count_mapping_safely(
        self,
        counts: dict[CardRank, int],
        rank_counts: dict[CardRank, int],
    ) -> None:
        for rank, amount in rank_counts.items():
            current_count = counts.get(rank, 0)
            counts[rank] = max(0, current_count - amount)

    def _build_state_matrix_from_known_unseen_counts(self, known_unseen_counts: Mapping[CardRank, int]) -> tuple[int, ...]:
        matrix: list[int] = []
        for rank, full_count in FULL_DECK_RANK_COUNTS.items():
            unseen_count = known_unseen_counts.get(rank, 0)
            index = 0
            while index < full_count:
                value = 1 if index < unseen_count else 0
                matrix.append(value)
                index += 1
        result = tuple(matrix)
        return result
