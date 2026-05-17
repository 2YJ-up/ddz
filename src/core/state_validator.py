from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.core.cards import CARD_LIMITS, assert_card_counts_within_deck, to_domain_rank
from src.core.rule_engine import RuleEngine
from src.core.visible_state import NEXT_POSITION, PLAYER_POSITIONS, VisibleGameState
from src.domain.actions import ActionType, CardAction, ClassifiedAction, PlayerSeat


@dataclass(frozen=True, slots=True)
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_visible_state(state: VisibleGameState, *, rule_engine: RuleEngine | None = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    engine = rule_engine or RuleEngine()

    _validate_positions(state, errors)
    _validate_card_lists(state, errors)
    _validate_num_cards_left(state, errors)
    _validate_public_history_counts(state, errors)
    _validate_history_actions(state, engine, errors, warnings)
    _validate_turn_order(state, warnings)

    if state.recognition_confidence < 0.85:
        warnings.append(
            f"recognition_confidence={state.recognition_confidence:.2f}; please confirm the recognized cards before acting"
        )
    if state.acting_player != state.self_position:
        warnings.append(
            f"acting_player={state.acting_player}; it is not currently self_position={state.self_position}'s turn"
        )

    return ValidationResult(errors=errors, warnings=warnings)


def require_valid_visible_state(state: VisibleGameState, *, rule_engine: RuleEngine | None = None) -> ValidationResult:
    result = validate_visible_state(state, rule_engine=rule_engine)
    if not result.ok:
        raise ValueError("; ".join(result.errors))
    return result


def _validate_positions(state: VisibleGameState, errors: list[str]) -> None:
    if state.self_position not in PLAYER_POSITIONS:
        errors.append(f"invalid self_position: {state.self_position}")
    if state.acting_player not in PLAYER_POSITIONS:
        errors.append(f"invalid acting_player: {state.acting_player}")
    for action in state.public_action_history:
        if action.player not in PLAYER_POSITIONS:
            errors.append(f"invalid action player: {action.player}")


def _validate_card_lists(state: VisibleGameState, errors: list[str]) -> None:
    try:
        visible_cards = list(state.self_hand)
        for action in state.public_action_history:
            visible_cards.extend(action.cards)
        assert_card_counts_within_deck(visible_cards)
    except ValueError as exc:
        errors.append(str(exc))

    if len(state.landlord_public_cards) > 3:
        errors.append("landlord_public_cards cannot contain more than 3 cards")
    try:
        assert_card_counts_within_deck(state.landlord_public_cards)
    except ValueError as exc:
        errors.append(f"invalid landlord_public_cards: {exc}")

    total_known_without_landlord_public = len(state.self_hand) + sum(len(action.cards) for action in state.public_action_history)
    if total_known_without_landlord_public > 54:
        errors.append("visible card total exceeds 54")


def _validate_num_cards_left(state: VisibleGameState, errors: list[str]) -> None:
    for player in PLAYER_POSITIONS:
        if player not in state.num_cards_left:
            errors.append(f"num_cards_left missing {player}")
        elif state.num_cards_left[player] < 0:
            errors.append(f"num_cards_left[{player}] cannot be negative")
        elif state.num_cards_left[player] > 20:
            errors.append(f"num_cards_left[{player}] cannot exceed 20")
    if state.self_position in state.num_cards_left and len(state.self_hand) != state.num_cards_left[state.self_position]:
        errors.append(
            f"self_hand has {len(state.self_hand)} cards but num_cards_left[{state.self_position}] is "
            f"{state.num_cards_left[state.self_position]}"
        )


def _validate_public_history_counts(state: VisibleGameState, errors: list[str]) -> None:
    played_by_player: Counter[str] = Counter()
    for action in state.public_action_history:
        if action.is_pass and len(action.cards) > 0:
            errors.append(f"pass action by {action.player} contains cards")
        if not action.is_pass and len(action.cards) == 0:
            errors.append(f"non-pass action by {action.player} has no cards")
        played_by_player[action.player] += len(action.cards)

    for player, played_count in played_by_player.items():
        opening_count = 20 if player == "landlord" else 17
        if played_count > opening_count:
            errors.append(f"{player} has publicly played {played_count} cards, above opening count {opening_count}")
        cards_left = state.num_cards_left.get(player)
        if cards_left is not None and played_count + cards_left > opening_count:
            errors.append(
                f"{player} played {played_count} cards and has {cards_left} left, above opening count {opening_count}"
            )


def _validate_history_actions(
    state: VisibleGameState,
    engine: RuleEngine,
    errors: list[str],
    warnings: list[str],
) -> None:
    previous_action: ClassifiedAction | None = None
    pass_count = 0
    for index, action in enumerate(state.public_action_history):
        card_action = CardAction(
            actor_seat=PlayerSeat.SELF,
            ranks=tuple(to_domain_rank(card) for card in action.cards),
            declared_type=ActionType.PASS if action.is_pass else ActionType.UNKNOWN,
        )
        classified = engine.classify_action(card_action)
        if classified.action_type is ActionType.INVALID:
            errors.append(f"public_action_history[{index}] is not a legal Doudizhu shape: {action.cards}")
        elif action.is_pass and previous_action is None:
            warnings.append(f"public_action_history[{index}] is a leading pass; confirm the history starts from this hand")
        elif not engine.is_legal_response(card_action, previous_action):
            errors.append(f"public_action_history[{index}] is not legal against the current trick")

        if not action.is_pass:
            previous_action = classified
            pass_count = 0
        elif previous_action is not None:
            pass_count += 1
            if pass_count >= 2:
                previous_action = None
                pass_count = 0


def _validate_turn_order(state: VisibleGameState, warnings: list[str]) -> None:
    if not state.public_action_history:
        return
    expected_next = NEXT_POSITION.get(state.public_action_history[0].player)
    for action in state.public_action_history[1:]:
        if expected_next is not None and action.player != expected_next:
            warnings.append("public_action_history turn order is not a continuous landlord -> down -> up cycle")
            return
        expected_next = NEXT_POSITION.get(action.player)
    if expected_next is not None and state.acting_player != expected_next:
        warnings.append(
            f"acting_player={state.acting_player} does not match next player inferred from history ({expected_next})"
        )
