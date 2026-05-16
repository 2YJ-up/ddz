from src.core.rule_engine import RuleEngine
from src.core.state_manager import StateManager
from src.core.state_normalizer import StateNormalizerConfig, StateObservationNormalizer
from src.domain.actions import ActionType, BoundingBox, CardDetection, PlayerSeat
from src.domain.cards import CardRank


def test_state_normalizer_caps_duplicate_visual_ranks() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    detections = tuple(
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=index * 10, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        )
        for index in range(6)
    )

    view = normalizer.normalize(detections)

    assert view.self_rank_counts[CardRank.THREE] == 4


def test_state_normalizer_waits_for_self_cards_before_starting() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(min_self_cards_to_start=1),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )

    waiting_view = normalizer.normalize(())

    assert waiting_view.hand_counts[PlayerSeat.SELF] == 0
    assert not normalizer.state_manager.is_started()


def test_state_normalizer_appends_self_action_when_cards_disappear() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    first_detections = (
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
        CardDetection(
            card_rank=CardRank.FOUR,
            confidence=0.9,
            bbox=BoundingBox(x=20, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )
    second_detections = (
        CardDetection(
            card_rank=CardRank.FOUR,
            confidence=0.9,
            bbox=BoundingBox(x=20, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )

    normalizer.normalize(first_detections)
    view = normalizer.normalize(second_detections)

    assert len(view.action_log) == 1
    assert view.action_log[0].action.ranks == (CardRank.THREE,)


def test_state_normalizer_waits_for_stable_bootstrap_frames() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(min_self_cards_to_start=1, bootstrap_stable_frames=2),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    detections = (
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )

    waiting_view = normalizer.normalize(detections)
    started_view = normalizer.normalize(detections)

    assert waiting_view.hand_counts[PlayerSeat.SELF] == 0
    assert started_view.self_rank_counts[CardRank.THREE] == 1


def test_state_normalizer_rebootstraps_when_more_cards_appear_before_actions() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(min_self_cards_to_start=1, bootstrap_stable_frames=1),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    partial_detections = (
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )
    fuller_detections = (
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
        CardDetection(
            card_rank=CardRank.FOUR,
            confidence=0.9,
            bbox=BoundingBox(x=20, y=100, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )

    normalizer.normalize(partial_detections)
    view = normalizer.normalize(fuller_detections)

    assert view.self_rank_counts[CardRank.THREE] == 1
    assert view.self_rank_counts[CardRank.FOUR] == 1
    assert len(view.action_log) == 0


def test_state_normalizer_applies_table_cards_as_current_trick_view() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(min_self_cards_to_start=1, bootstrap_stable_frames=1),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    detections = (
        CardDetection(
            card_rank=CardRank.THREE,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=700, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
        CardDetection(
            card_rank=CardRank.ACE,
            confidence=0.9,
            bbox=BoundingBox(x=400, y=300, width=8, height=12),
            seat_region=PlayerSeat.TABLE,
        ),
        CardDetection(
            card_rank=CardRank.ACE,
            confidence=0.9,
            bbox=BoundingBox(x=420, y=300, width=8, height=12),
            seat_region=PlayerSeat.TABLE,
        ),
    )

    view = normalizer.normalize(detections)

    assert view.current_turn is PlayerSeat.SELF
    assert view.current_trick_action is not None
    assert view.current_trick_action.action_type is ActionType.PAIR
    assert view.public_played_counts[CardRank.ACE] == 2
    assert view.known_unseen_counts[CardRank.THREE] == 3
    assert view.known_unseen_counts[CardRank.ACE] == 2
    assert sum(view.state_matrix) == 51


def test_state_normalizer_live_self_observation_overrides_stale_hand_view() -> None:
    normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(min_self_cards_to_start=1, bootstrap_stable_frames=1),
        state_manager=StateManager(rule_engine=RuleEngine()),
    )
    stale_detections = tuple(
        CardDetection(
            card_rank=CardRank.JACK,
            confidence=0.9,
            bbox=BoundingBox(x=index * 10, y=700, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        )
        for index in range(4)
    )
    current_detections = (
        CardDetection(
            card_rank=CardRank.FOUR,
            confidence=0.9,
            bbox=BoundingBox(x=10, y=700, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
        CardDetection(
            card_rank=CardRank.ACE,
            confidence=0.9,
            bbox=BoundingBox(x=20, y=700, width=8, height=12),
            seat_region=PlayerSeat.SELF,
        ),
    )

    normalizer.normalize(stale_detections)
    normalizer.state_manager._current_turn = PlayerSeat.LEFT_OPPONENT
    view = normalizer.normalize(current_detections)

    assert CardRank.JACK not in view.self_rank_counts
    assert view.self_rank_counts[CardRank.FOUR] == 1
    assert view.self_rank_counts[CardRank.ACE] == 1
