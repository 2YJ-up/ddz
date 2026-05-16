from pathlib import Path

from src.ai.rl_forward import RlForwardConfig, RlForwardService
from src.core.rule_engine import RuleEngine
from src.domain.actions import GameStateView, PlayerSeat
from src.domain.cards import CardRank


def test_rl_forward_uses_self_rank_counts_for_fallback() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=2),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.THREE: 1, CardRank.FOUR: 2},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 3, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert len(recommendations) == 2
    assert recommendations[0].reason_code == "heuristic_fallback"
