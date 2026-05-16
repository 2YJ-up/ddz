from src.core.rule_engine import RuleEngine
from src.domain.actions import ActionType, CardAction, PlayerSeat
from src.domain.cards import CardRank


def make_action(*ranks: CardRank) -> CardAction:
    action = CardAction(actor_seat=PlayerSeat.SELF, ranks=tuple(ranks))
    return action


def test_classifies_basic_and_sequence_actions() -> None:
    engine = RuleEngine()

    single = engine.classify_action(make_action(CardRank.THREE))
    pair = engine.classify_action(make_action(CardRank.FOUR, CardRank.FOUR))
    triple_with_pair = engine.classify_action(
        make_action(CardRank.FIVE, CardRank.FIVE, CardRank.FIVE, CardRank.SIX, CardRank.SIX)
    )
    straight = engine.classify_action(
        make_action(CardRank.THREE, CardRank.FOUR, CardRank.FIVE, CardRank.SIX, CardRank.SEVEN)
    )
    pair_straight = engine.classify_action(
        make_action(CardRank.THREE, CardRank.THREE, CardRank.FOUR, CardRank.FOUR, CardRank.FIVE, CardRank.FIVE)
    )

    assert single.action_type is ActionType.SINGLE
    assert pair.action_type is ActionType.PAIR
    assert triple_with_pair.action_type is ActionType.TRIPLE_WITH_PAIR
    assert straight.action_type is ActionType.STRAIGHT
    assert pair_straight.action_type is ActionType.PAIR_STRAIGHT


def test_bomb_and_rocket_comparison() -> None:
    engine = RuleEngine()

    straight = engine.classify_action(
        make_action(CardRank.THREE, CardRank.FOUR, CardRank.FIVE, CardRank.SIX, CardRank.SEVEN)
    )
    bomb = engine.classify_action(make_action(CardRank.NINE, CardRank.NINE, CardRank.NINE, CardRank.NINE))
    bigger_bomb = engine.classify_action(make_action(CardRank.TEN, CardRank.TEN, CardRank.TEN, CardRank.TEN))
    rocket = engine.classify_action(make_action(CardRank.SMALL_JOKER, CardRank.BIG_JOKER))

    assert bomb.action_type is ActionType.BOMB
    assert rocket.action_type is ActionType.ROCKET
    assert engine.compare_actions(bomb, straight)
    assert engine.compare_actions(bigger_bomb, bomb)
    assert engine.compare_actions(rocket, bigger_bomb)


def test_enumerates_legal_responses_against_pair() -> None:
    engine = RuleEngine()
    previous = engine.classify_action(make_action(CardRank.SEVEN, CardRank.SEVEN))
    hand_counts = {
        CardRank.SIX: 2,
        CardRank.EIGHT: 2,
        CardRank.THREE: 4,
        CardRank.SMALL_JOKER: 1,
        CardRank.BIG_JOKER: 1,
    }

    legal = engine.enumerate_legal_actions(
        hand_counts=hand_counts,
        actor_seat=PlayerSeat.SELF,
        previous_action=previous,
    )
    legal_types = tuple(engine.classify_action(action).action_type for action in legal)
    legal_ranks = tuple(action.ranks for action in legal)

    assert ActionType.PASS in legal_types
    assert (CardRank.EIGHT, CardRank.EIGHT) in legal_ranks
    assert (CardRank.THREE, CardRank.THREE, CardRank.THREE, CardRank.THREE) in legal_ranks
    assert (CardRank.SMALL_JOKER, CardRank.BIG_JOKER) in legal_ranks
    assert (CardRank.SIX, CardRank.SIX) not in legal_ranks
