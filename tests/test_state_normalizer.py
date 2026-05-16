from src.core.rule_engine import RuleEngine
from src.core.state_manager import StateManager
from src.core.state_normalizer import StateNormalizerConfig, StateObservationNormalizer
from src.domain.actions import BoundingBox, CardDetection, PlayerSeat
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
