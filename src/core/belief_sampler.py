from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from src.core.cards import (
    CARD_RANKS,
    count_cards,
    make_full_deck,
    normalize_cards,
    subtract_cards,
    to_domain_counts,
    to_domain_rank,
)
from src.core.rule_engine import RuleEngine
from src.core.visible_state import PLAYER_POSITIONS, VisibleAction, VisibleGameState
from src.domain.actions import ActionType, CardAction, PlayerSeat


@dataclass(frozen=True, slots=True)
class BeliefSample:
    hands: dict[str, list[str]]
    weight: float
    reasons: list[str] = field(default_factory=list)


def build_unknown_pool(state: VisibleGameState) -> list[str]:
    pool = make_full_deck()
    public_cards: list[str] = []
    for action in state.public_action_history:
        public_cards.extend(action.cards)
    pool = subtract_cards(pool, state.self_hand)
    pool = subtract_cards(pool, public_cards)
    fixed_landlord_cards = _remaining_public_landlord_cards(state)
    if state.self_position != "landlord" and fixed_landlord_cards:
        pool = subtract_cards(pool, fixed_landlord_cards)
    return pool


def sample_hidden_hands(
    state: VisibleGameState,
    n_samples: int = 256,
    use_pass_constraints: bool = True,
    seed: int | None = None,
) -> list[BeliefSample]:
    opponents = [player for player in PLAYER_POSITIONS if player != state.self_position]
    fixed_hands = _fixed_known_hands(state)
    pool = build_unknown_pool(state)
    required_counts: dict[str, int] = {}
    for player in opponents:
        fixed_count = len(fixed_hands.get(player, []))
        required = int(state.num_cards_left[player]) - fixed_count
        if required < 0:
            raise ValueError(f"fixed known cards exceed num_cards_left for {player}")
        required_counts[player] = required

    required_total = sum(required_counts.values())
    if required_total != len(pool):
        raise ValueError(f"unknown pool has {len(pool)} cards but opponents require {required_total}")

    rng = random.Random(seed)
    samples: list[BeliefSample] = []
    sample_count = max(0, int(n_samples))
    for _ in range(sample_count):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        cursor = 0
        hands: dict[str, list[str]] = {player: list(fixed_hands.get(player, [])) for player in opponents}
        for player in opponents:
            take = required_counts[player]
            hands[player].extend(shuffled[cursor : cursor + take])
            hands[player] = normalize_cards(hands[player])
            cursor += take
        weight, reasons = _weight_sample(state, hands, use_pass_constraints=use_pass_constraints)
        samples.append(BeliefSample(hands=hands, weight=weight, reasons=reasons))
    return samples


def _fixed_known_hands(state: VisibleGameState) -> dict[str, list[str]]:
    fixed: dict[str, list[str]] = {}
    if state.self_position != "landlord":
        landlord_cards = _remaining_public_landlord_cards(state)
        if landlord_cards:
            fixed["landlord"] = landlord_cards
    return fixed


def _remaining_public_landlord_cards(state: VisibleGameState) -> list[str]:
    remaining = count_cards(state.landlord_public_cards)
    landlord_played: list[str] = []
    for action in state.public_action_history:
        if action.player == "landlord":
            landlord_played.extend(action.cards)
    for card, amount in count_cards(landlord_played).items():
        current = remaining.get(card, 0)
        remaining[card] = max(0, current - amount)
        if remaining[card] == 0:
            remaining.pop(card, None)
    cards: list[str] = []
    for rank in CARD_RANKS:
        cards.extend([rank] * remaining.get(rank, 0))
    return cards


def _weight_sample(
    state: VisibleGameState,
    hands: dict[str, list[str]],
    *,
    use_pass_constraints: bool,
) -> tuple[float, list[str]]:
    weight = 1.0
    reasons: list[str] = []
    if use_pass_constraints:
        pass_weight, pass_reasons = _pass_constraint_weight(state, hands)
        weight *= pass_weight
        reasons.extend(pass_reasons)
    for player, cards in hands.items():
        if len(cards) <= 3 and _can_finish_in_one_move(cards):
            reasons.append(f"{player} may be able to finish in one move")
            weight *= 1.05
    return max(0.001, weight), reasons


def _pass_constraint_weight(state: VisibleGameState, hands: dict[str, list[str]]) -> tuple[float, list[str]]:
    weight = 1.0
    reasons: list[str] = []
    engine = RuleEngine()
    last_non_pass: VisibleAction | None = None
    consecutive_passes: Counter[str] = Counter()
    for action in state.public_action_history:
        if not action.is_pass:
            last_non_pass = action
            consecutive_passes[action.player] = 0
        else:
            consecutive_passes[action.player] += 1
            can_apply_pass_constraint = action.player in hands and last_non_pass is not None
            if can_apply_pass_constraint and _hand_can_beat(hands[action.player], last_non_pass.cards, engine):
                weight *= 0.72
                reasons.append(f"{action.player} passed despite possible responses")
            if can_apply_pass_constraint and consecutive_passes[action.player] >= 2:
                weight *= 0.9
                reasons.append(f"{action.player} had consecutive passes")
    return weight, reasons


def _hand_can_beat(hand: list[str], previous_cards: list[str], engine: RuleEngine) -> bool:
    previous = engine.classify_action(
        CardAction(actor_seat=PlayerSeat.RIGHT_OPPONENT, ranks=tuple(to_domain_rank(card) for card in previous_cards))
    )
    legal = engine.enumerate_legal_actions(
        hand_counts=to_domain_counts(hand),
        actor_seat=PlayerSeat.SELF,
        previous_action=previous,
    )
    return any(not action.is_pass() for action in legal)


def _can_finish_in_one_move(cards: list[str]) -> bool:
    if not cards:
        return False
    engine = RuleEngine()
    action = CardAction(actor_seat=PlayerSeat.SELF, ranks=tuple(to_domain_rank(card) for card in cards))
    return engine.classify_action(action).action_type not in {ActionType.INVALID, ActionType.PASS}
