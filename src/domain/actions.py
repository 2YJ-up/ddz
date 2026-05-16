from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from src.domain.cards import CardRank, normalize_ranks


class PlayerSeat(str, Enum):
    SELF = "self"
    LEFT_OPPONENT = "left_opponent"
    RIGHT_OPPONENT = "right_opponent"
    TABLE = "table"


NEXT_SEAT: Mapping[PlayerSeat, PlayerSeat] = {
    PlayerSeat.SELF: PlayerSeat.LEFT_OPPONENT,
    PlayerSeat.LEFT_OPPONENT: PlayerSeat.RIGHT_OPPONENT,
    PlayerSeat.RIGHT_OPPONENT: PlayerSeat.SELF,
}


class ActionType(str, Enum):
    UNKNOWN = "unknown"
    INVALID = "invalid"
    PASS = "pass"
    SINGLE = "single"
    PAIR = "pair"
    TRIPLE = "triple"
    TRIPLE_WITH_SINGLE = "triple_with_single"
    TRIPLE_WITH_PAIR = "triple_with_pair"
    STRAIGHT = "straight"
    PAIR_STRAIGHT = "pair_straight"
    TRIPLE_STRAIGHT = "triple_straight"
    AIRPLANE_WITH_SINGLES = "airplane_with_singles"
    AIRPLANE_WITH_PAIRS = "airplane_with_pairs"
    FOUR_WITH_TWO_SINGLES = "four_with_two_singles"
    FOUR_WITH_TWO_PAIRS = "four_with_two_pairs"
    BOMB = "bomb"
    ROCKET = "rocket"


@dataclass(frozen=True, slots=True)
class CardAction:
    actor_seat: PlayerSeat
    ranks: tuple[CardRank, ...] = field(default_factory=tuple)
    declared_type: ActionType = ActionType.UNKNOWN
    observed_at_ms: int = 0
    source_frame_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", normalize_ranks(self.ranks))

    def card_count(self) -> int:
        count = len(self.ranks)
        return count

    def is_pass(self) -> bool:
        result = self.card_count() == 0
        return result


@dataclass(frozen=True, slots=True)
class ClassifiedAction:
    action_type: ActionType
    primary_rank: CardRank | None
    card_count: int
    chain_length: int = 0

    def is_valid(self) -> bool:
        result = self.action_type is not ActionType.INVALID
        return result


@dataclass(frozen=True, slots=True)
class InitialDeal:
    self_cards: tuple[CardRank, ...]
    landlord_cards: tuple[CardRank, ...]
    landlord_seat: PlayerSeat
    first_turn: PlayerSeat

    def __post_init__(self) -> None:
        object.__setattr__(self, "self_cards", normalize_ranks(self.self_cards))
        object.__setattr__(self, "landlord_cards", normalize_ranks(self.landlord_cards))


@dataclass(frozen=True, slots=True)
class ActionLogEntry:
    action: CardAction
    sequence_index: int


@dataclass(frozen=True, slots=True)
class GameStateView:
    current_turn: PlayerSeat
    landlord_seat: PlayerSeat
    action_log: tuple[ActionLogEntry, ...]
    current_trick_action: ClassifiedAction | None
    self_rank_counts: Mapping[CardRank, int]
    public_played_counts: Mapping[CardRank, int]
    known_unseen_counts: Mapping[CardRank, int]
    hand_counts: Mapping[PlayerSeat, int]
    state_matrix: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActionRecommendation:
    action: CardAction
    probability: float
    expected_win_rate: float
    reason_code: str
    reason_text: str = ""
    risk_text: str = ""
    action_label: str = ""


@dataclass(frozen=True, slots=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class FrameBuffer:
    pixels: bytes
    width: int
    height: int
    timestamp_ms: int
    window_rect: WindowRect
    frame_id: str


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CardDetection:
    card_rank: CardRank
    confidence: float
    bbox: BoundingBox
    seat_region: PlayerSeat


@dataclass(frozen=True, slots=True)
class RenderFrame:
    text_blocks: tuple[str, ...]
    highlight_ranks: tuple[CardRank, ...]
    anchor_rect: WindowRect
