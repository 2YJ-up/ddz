from src.core.rule_engine import RuleEngine
from src.core.state_manager import StateManager
from src.domain.actions import CardAction, InitialDeal, PlayerSeat
from src.domain.cards import CardRank


def make_manager() -> StateManager:
    manager = StateManager(rule_engine=RuleEngine())
    deal = InitialDeal(
        self_cards=(
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
            CardRank.THREE,
            CardRank.FOUR,
            CardRank.FIVE,
            CardRank.SIX,
        ),
        landlord_cards=(CardRank.SEVEN, CardRank.EIGHT, CardRank.NINE),
        landlord_seat=PlayerSeat.SELF,
        first_turn=PlayerSeat.SELF,
    )
    manager.start_hand(deal)
    return manager


def test_state_manager_replays_base_truths_into_view() -> None:
    manager = make_manager()

    manager.append_action(CardAction(actor_seat=PlayerSeat.SELF, ranks=(CardRank.THREE,)))
    manager.append_action(CardAction(actor_seat=PlayerSeat.LEFT_OPPONENT, ranks=()))
    manager.append_action(CardAction(actor_seat=PlayerSeat.RIGHT_OPPONENT, ranks=()))

    view = manager.build_view()
    exported = manager.export_base_truths()

    assert view.current_turn is PlayerSeat.SELF
    assert view.current_trick_action is None
    assert view.self_rank_counts[CardRank.THREE] == 1
    assert view.public_played_counts[CardRank.THREE] == 1
    assert view.known_unseen_counts[CardRank.THREE] == 2
    assert view.hand_counts[PlayerSeat.SELF] == 19
    assert len(view.state_matrix) == 54
    assert sum(view.state_matrix) == 34
    assert set(exported.keys()) == {"initial_deal", "current_turn", "action_log"}


def test_state_manager_rejects_wrong_turn_action() -> None:
    manager = make_manager()

    try:
        manager.append_action(CardAction(actor_seat=PlayerSeat.LEFT_OPPONENT, ranks=(CardRank.THREE,)))
        raised = False
    except ValueError:
        raised = True

    assert raised
