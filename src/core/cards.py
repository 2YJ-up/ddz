from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Mapping

from src.domain.cards import CardRank

CardCode = Literal["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "X", "D"]

CARD_RANKS: tuple[str, ...] = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "X", "D")
SEQUENCE_CARD_RANKS: tuple[str, ...] = ("3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
CARD_LIMITS: Mapping[str, int] = {
    "3": 4,
    "4": 4,
    "5": 4,
    "6": 4,
    "7": 4,
    "8": 4,
    "9": 4,
    "10": 4,
    "J": 4,
    "Q": 4,
    "K": 4,
    "A": 4,
    "2": 4,
    "X": 1,
    "D": 1,
}

PERFECTDOU_CARD_MAP: Mapping[str, int] = {
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
    "2": 17,
    "X": 20,
    "D": 30,
}

_CARD_ORDER: Mapping[str, int] = {rank: index for index, rank in enumerate(CARD_RANKS)}
_TO_DOMAIN_RANK: Mapping[str, CardRank] = {
    "3": CardRank.THREE,
    "4": CardRank.FOUR,
    "5": CardRank.FIVE,
    "6": CardRank.SIX,
    "7": CardRank.SEVEN,
    "8": CardRank.EIGHT,
    "9": CardRank.NINE,
    "10": CardRank.TEN,
    "J": CardRank.JACK,
    "Q": CardRank.QUEEN,
    "K": CardRank.KING,
    "A": CardRank.ACE,
    "2": CardRank.TWO,
    "X": CardRank.SMALL_JOKER,
    "D": CardRank.BIG_JOKER,
}
_FROM_DOMAIN_RANK: Mapping[CardRank, str] = {value: key for key, value in _TO_DOMAIN_RANK.items()}


def card_sort_key(card: str) -> int:
    return _CARD_ORDER[normalize_card(card)]


def normalize_card(card: str) -> str:
    normalized = str(card).strip().upper()
    aliases = {
        "T": "10",
        "SJ": "X",
        "SMALL_JOKER": "X",
        "SMALLJOKER": "X",
        "JOKER_SMALL": "X",
        "BJ": "D",
        "BIG_JOKER": "D",
        "BIGJOKER": "D",
        "JOKER_BIG": "D",
        "LITTLE_JOKER": "X",
        "BLACK_JOKER": "X",
        "RED_JOKER": "D",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CARD_LIMITS:
        raise ValueError(f"unknown card rank: {card!r}")
    return normalized


def normalize_cards(cards: Iterable[str] | None) -> list[str]:
    if cards is None:
        return []
    normalized = [normalize_card(card) for card in cards]
    normalized.sort(key=card_sort_key)
    return normalized


def make_full_deck() -> list[str]:
    deck: list[str] = []
    for rank in CARD_RANKS:
        deck.extend([rank] * int(CARD_LIMITS[rank]))
    return deck


def count_cards(cards: Iterable[str]) -> Counter[str]:
    return Counter(normalize_card(card) for card in cards)


def assert_card_counts_within_deck(cards: Iterable[str]) -> None:
    counts = count_cards(cards)
    for rank, amount in counts.items():
        if amount > CARD_LIMITS[rank]:
            raise ValueError(f"rank {rank} appears {amount} times, above deck limit {CARD_LIMITS[rank]}")


def subtract_cards(pool: Iterable[str], cards_to_remove: Iterable[str]) -> list[str]:
    counts = count_cards(pool)
    for card, amount in count_cards(cards_to_remove).items():
        next_amount = counts.get(card, 0) - amount
        if next_amount < 0:
            raise ValueError(f"cannot remove {amount} of {card} from card pool")
        if next_amount == 0:
            counts.pop(card, None)
        else:
            counts[card] = next_amount
    result: list[str] = []
    for rank in CARD_RANKS:
        result.extend([rank] * counts.get(rank, 0))
    return result


def to_domain_rank(card: str) -> CardRank:
    return _TO_DOMAIN_RANK[normalize_card(card)]


def from_domain_rank(rank: CardRank) -> str:
    return _FROM_DOMAIN_RANK[rank]


def to_domain_counts(cards: Iterable[str]) -> dict[CardRank, int]:
    counts: dict[CardRank, int] = {}
    for card, amount in count_cards(cards).items():
        counts[to_domain_rank(card)] = int(amount)
    return counts


def from_domain_ranks(ranks: Iterable[CardRank]) -> list[str]:
    cards = [from_domain_rank(rank) for rank in ranks]
    cards.sort(key=card_sort_key)
    return cards


def unique_sorted(cards: Iterable[str]) -> list[str]:
    return sorted(set(normalize_cards(cards)), key=card_sort_key)
