from __future__ import annotations

from dataclasses import dataclass

from src.core.state_manager import StateManager
from src.domain.actions import CardAction, CardDetection, GameStateView, InitialDeal, PlayerSeat
from src.domain.cards import CardRank, FULL_DECK_RANK_COUNTS


@dataclass(frozen=True, slots=True)
class StateNormalizerConfig:
    landlord_seat: PlayerSeat = PlayerSeat.SELF
    first_turn: PlayerSeat = PlayerSeat.SELF
    bootstrap_from_self_detections: bool = True


class StateObservationNormalizer:
    def __init__(self, config: StateNormalizerConfig, state_manager: StateManager) -> None:
        self.config = config
        self.state_manager = state_manager

    def normalize(self, detections: tuple[CardDetection, ...]) -> GameStateView:
        if not self.state_manager.is_started():
            self._start_observed_hand(detections)
        view = self.state_manager.build_view()
        return view

    def start_hand(self, initial_deal: InitialDeal) -> None:
        self.state_manager.start_hand(initial_deal)

    def append_action(self, action: CardAction) -> None:
        self.state_manager.append_action(action)

    def _start_observed_hand(self, detections: tuple[CardDetection, ...]) -> None:
        self_cards = ()
        if self.config.bootstrap_from_self_detections:
            self_cards = self._extract_self_cards(detections)
        initial_deal = InitialDeal(
            self_cards=self_cards,
            landlord_cards=(),
            landlord_seat=self.config.landlord_seat,
            first_turn=self.config.first_turn,
        )
        self.state_manager.start_hand(initial_deal)

    def _extract_self_cards(self, detections: tuple[CardDetection, ...]) -> tuple[CardRank, ...]:
        ordered = tuple(sorted(detections, key=lambda item: (item.bbox.x, item.bbox.y, int(item.card_rank))))
        ranks = []
        rank_counts: dict[CardRank, int] = {}
        for detection in ordered:
            if detection.seat_region is PlayerSeat.SELF:
                current_count = rank_counts.get(detection.card_rank, 0)
                allowed_count = FULL_DECK_RANK_COUNTS[detection.card_rank]
                if current_count < allowed_count:
                    ranks.append(detection.card_rank)
                    rank_counts[detection.card_rank] = current_count + 1
        result = tuple(ranks)
        return result
