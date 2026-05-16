from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterable, Iterator, Mapping


class Suit(str, Enum):
    SPADE = "S"
    HEART = "H"
    CLUB = "C"
    DIAMOND = "D"
    JOKER = "J"


class CardRank(IntEnum):
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15
    SMALL_JOKER = 16
    BIG_JOKER = 17


STANDARD_RANKS: tuple[CardRank, ...] = (
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

SEQUENCE_RANKS: tuple[CardRank, ...] = (
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

JOKER_RANKS: tuple[CardRank, ...] = (
    CardRank.SMALL_JOKER,
    CardRank.BIG_JOKER,
)

FULL_DECK_RANK_COUNTS: Mapping[CardRank, int] = {
    CardRank.THREE: 4,
    CardRank.FOUR: 4,
    CardRank.FIVE: 4,
    CardRank.SIX: 4,
    CardRank.SEVEN: 4,
    CardRank.EIGHT: 4,
    CardRank.NINE: 4,
    CardRank.TEN: 4,
    CardRank.JACK: 4,
    CardRank.QUEEN: 4,
    CardRank.KING: 4,
    CardRank.ACE: 4,
    CardRank.TWO: 4,
    CardRank.SMALL_JOKER: 1,
    CardRank.BIG_JOKER: 1,
}


@dataclass(frozen=True, slots=True)
class Card:
    rank: CardRank
    suit: Suit

    def __post_init__(self) -> None:
        joker_rank = self.rank in JOKER_RANKS
        joker_suit = self.suit is Suit.JOKER
        if joker_rank != joker_suit:
            raise ValueError("joker rank and joker suit must match")

    def label(self) -> str:
        label = f"{self.rank.name}:{self.suit.value}"
        return label

    def sort_key(self) -> tuple[int, str]:
        key = (int(self.rank), self.suit.value)
        return key


class Deck:
    def __init__(self, cards: Iterable[Card]) -> None:
        self.cards = tuple(sorted(tuple(cards), key=lambda card: card.sort_key()))

    def __iter__(self) -> Iterator[Card]:
        iterator = iter(self.cards)
        return iterator

    def __len__(self) -> int:
        length = len(self.cards)
        return length

    def rank_counts(self) -> dict[CardRank, int]:
        counts = dict(Counter(card.rank for card in self.cards))
        return counts

    @staticmethod
    def standard() -> "Deck":
        cards: list[Card] = []
        suits = (Suit.SPADE, Suit.HEART, Suit.CLUB, Suit.DIAMOND)
        for rank in STANDARD_RANKS:
            for suit in suits:
                cards.append(Card(rank=rank, suit=suit))
        cards.append(Card(rank=CardRank.SMALL_JOKER, suit=Suit.JOKER))
        cards.append(Card(rank=CardRank.BIG_JOKER, suit=Suit.JOKER))
        deck = Deck(cards)
        return deck


def normalize_ranks(ranks: Iterable[CardRank]) -> tuple[CardRank, ...]:
    normalized = tuple(sorted(tuple(ranks), key=int))
    return normalized


def count_ranks(ranks: Iterable[CardRank]) -> dict[CardRank, int]:
    counts = dict(Counter(tuple(ranks)))
    return counts
