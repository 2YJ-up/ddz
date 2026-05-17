from pathlib import Path

import numpy as np

from src.ai.perfectdou_adapter import (
    PERFECTDOU_ACTION_ID_COLUMN,
    PERFECTDOU_ACTION_OFFSET,
    PERFECTDOU_ACTION_STRIDE,
    PERFECTDOU_ACTION_VALID_COLUMN,
    PERFECTDOU_INPUT_SIZE,
    PerfectDouAdapter,
)
from src.ai.rl_forward import RlForwardConfig, RlForwardService
from src.core.rule_engine import RuleEngine
from src.domain.actions import CardAction, GameStateView, PlayerSeat
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


class InvalidShapeSession:
    def run(self, output_names, input_feed):
        raise Exception("InvalidArgument: Got invalid dimensions for input")


class FixedPolicySession:
    def __init__(self, preferred_index: int) -> None:
        self.preferred_index = preferred_index
        self.last_input = None

    def run(self, output_names, input_feed):
        self.last_input = next(iter(input_feed.values()))
        scores = np.full((1, 621), -3.4028235e38, dtype=np.float32)
        scores[0, self.preferred_index] = 9.0
        return [scores]


def test_perfectdou_action_space_matches_official_indices() -> None:
    adapter = PerfectDouAdapter(RuleEngine())

    assert len(adapter.action_space) == 621
    assert adapter.action_space["3"] == 0
    assert adapter.action_space["333444**"] == 512
    assert adapter.action_space["3333****"] == 593
    assert adapter.action_space["BR"] == 619
    assert adapter.action_space["pass"] == 620


def test_rl_forward_builds_perfectdou_policy_tensor() -> None:
    rule_engine = RuleEngine()
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=rule_engine,
    )
    service._expected_input_size = PERFECTDOU_INPUT_SIZE
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.THREE: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )
    actions = (CardAction(actor_seat=PlayerSeat.SELF, ranks=(CardRank.THREE,)),)

    tensor = service._build_policy_tensor(view, actions)

    assert tensor.shape == (1, PERFECTDOU_INPUT_SIZE)
    assert tensor[0, PERFECTDOU_ACTION_OFFSET + PERFECTDOU_ACTION_ID_COLUMN] == 0.0
    assert tensor[0, PERFECTDOU_ACTION_OFFSET + PERFECTDOU_ACTION_VALID_COLUMN] == 1.0
    assert np.sum(tensor[0, PERFECTDOU_ACTION_OFFSET + PERFECTDOU_ACTION_STRIDE :]) == 0.0


def test_rl_forward_uses_perfectdou_logits_for_action_ranking() -> None:
    rule_engine = RuleEngine()
    adapter = PerfectDouAdapter(rule_engine)
    preferred_action = CardAction(actor_seat=PlayerSeat.SELF, ranks=(CardRank.FOUR,))
    preferred_index = adapter.action_index_for_action(preferred_action)
    assert preferred_index is not None
    session = FixedPolicySession(preferred_index)
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1, input_name="input", output_name="action_logit"),
        rule_engine=rule_engine,
    )
    service._session = session
    service._model_loaded = True
    service._expected_input_size = PERFECTDOU_INPUT_SIZE
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.THREE: 1, CardRank.FOUR: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 2, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert recommendations[0].action.ranks == (CardRank.FOUR,)
    assert recommendations[0].reason_code == "onnx_policy"
    assert session.last_input.shape == (1, PERFECTDOU_INPUT_SIZE)


def test_rl_forward_falls_back_when_policy_dimension_mismatches(capsys) -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1, input_name="input", output_name="output"),
        rule_engine=RuleEngine(),
    )
    service._session = InvalidShapeSession()
    service._model_loaded = True
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.THREE: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)
    captured = capsys.readouterr()

    assert len(recommendations) == 1
    assert recommendations[0].reason_code == "heuristic_fallback"
    assert "Model dimension mismatch, falling back to heuristics." in captured.out


def test_rl_forward_falls_back_when_state_encoder_size_is_not_model_size(capsys) -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    service._session = InvalidShapeSession()
    service._model_loaded = True
    service._expected_input_size = 32738
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.THREE: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)
    captured = capsys.readouterr()

    assert len(recommendations) == 1
    assert recommendations[0].reason_code == "heuristic_fallback"
    assert "Model dimension mismatch, falling back to heuristics." in captured.out


def test_rl_forward_recommends_for_self_even_when_turn_pointer_is_stale() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=2),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.LEFT_OPPONENT,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.KING: 4, CardRank.THREE: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 5, PlayerSeat.LEFT_OPPONENT: 1, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert len(recommendations) == 2
    assert recommendations[0].action.actor_seat is PlayerSeat.SELF
    assert recommendations[0].action.card_count() > 0


def test_rl_forward_keeps_pass_win_rate_low_when_pass_is_only_action() -> None:
    rule_engine = RuleEngine()
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=rule_engine,
    )
    service.load()
    previous_action = rule_engine.classify_action(
        CardAction(actor_seat=PlayerSeat.RIGHT_OPPONENT, ranks=(CardRank.ACE, CardRank.ACE))
    )
    view = GameStateView(
        current_turn=PlayerSeat.LEFT_OPPONENT,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=previous_action,
        self_rank_counts={CardRank.THREE: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 3, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert len(recommendations) == 1
    assert recommendations[0].action.is_pass()
    assert recommendations[0].expected_win_rate <= 0.05


def test_rl_forward_prefers_small_single_over_bomb_for_small_table_single() -> None:
    rule_engine = RuleEngine()
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=3),
        rule_engine=rule_engine,
    )
    service.load()
    previous_action = rule_engine.classify_action(CardAction(actor_seat=PlayerSeat.RIGHT_OPPONENT, ranks=(CardRank.THREE,)))
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=previous_action,
        self_rank_counts={CardRank.FOUR: 1, CardRank.JACK: 4, CardRank.SMALL_JOKER: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 6, PlayerSeat.LEFT_OPPONENT: 13, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert recommendations[0].action.ranks == (CardRank.FOUR,)
    assert recommendations[0].reason_text != ""


def test_rl_forward_outputs_risk_text_for_unseen_bomb() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.FOUR: 1},
        public_played_counts={CardRank.THREE: 1},
        known_unseen_counts={CardRank.SEVEN: 4, CardRank.SMALL_JOKER: 1, CardRank.BIG_JOKER: 1},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 13, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert "7" in recommendations[0].risk_text
    assert "王炸" in recommendations[0].risk_text


def test_rl_forward_suppresses_risk_text_without_public_history() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.FOUR: 1},
        public_played_counts={},
        known_unseen_counts={CardRank.SEVEN: 4, CardRank.SMALL_JOKER: 1, CardRank.BIG_JOKER: 1},
        hand_counts={PlayerSeat.SELF: 1, PlayerSeat.LEFT_OPPONENT: 13, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert recommendations[0].risk_text == ""


def test_rl_forward_counts_unseen_bomb_and_rocket_control_risk() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.JACK: 4},
        public_played_counts={CardRank.THREE: 1},
        known_unseen_counts={
            CardRank.KING: 4,
            CardRank.SMALL_JOKER: 1,
            CardRank.BIG_JOKER: 1,
        },
        hand_counts={PlayerSeat.SELF: 4, PlayerSeat.LEFT_OPPONENT: 13, PlayerSeat.RIGHT_OPPONENT: 14},
        state_matrix=tuple(1 for _ in range(54)),
    )
    classified = service.rule_engine.classify_action(
        CardAction(actor_seat=PlayerSeat.SELF, ranks=(CardRank.JACK, CardRank.JACK, CardRank.JACK, CardRank.JACK))
    )

    bomb_risk = service._unseen_bomb_risk_count(view)
    higher_risk = service._higher_bomb_or_rocket_risk(classified, view)

    assert bomb_risk == 2
    assert higher_risk == 2


def test_rl_forward_min_turn_estimate_treats_straight_as_one_turn() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    hand_counts = {
        CardRank.THREE: 1,
        CardRank.FOUR: 1,
        CardRank.FIVE: 1,
        CardRank.SIX: 1,
        CardRank.SEVEN: 1,
    }

    turn_count = service._turn_count_estimate(hand_counts)

    assert turn_count == 1


def test_rl_forward_prefers_straight_when_leading_small_sequence_hand() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={
            CardRank.THREE: 1,
            CardRank.FOUR: 1,
            CardRank.FIVE: 1,
            CardRank.SIX: 1,
            CardRank.SEVEN: 1,
        },
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 5, PlayerSeat.LEFT_OPPONENT: 17, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert recommendations[0].action.ranks == (
        CardRank.THREE,
        CardRank.FOUR,
        CardRank.FIVE,
        CardRank.SIX,
        CardRank.SEVEN,
    )
    assert recommendations[0].action_label == "顺子 3-7"


def test_rl_forward_labels_joker_rocket_in_chinese() -> None:
    service = RlForwardService(
        config=RlForwardConfig(model_path=Path("models/missing.onnx"), top_n=1),
        rule_engine=RuleEngine(),
    )
    service.load()
    view = GameStateView(
        current_turn=PlayerSeat.SELF,
        landlord_seat=PlayerSeat.SELF,
        action_log=(),
        current_trick_action=None,
        self_rank_counts={CardRank.SMALL_JOKER: 1, CardRank.BIG_JOKER: 1},
        public_played_counts={},
        known_unseen_counts={},
        hand_counts={PlayerSeat.SELF: 2, PlayerSeat.LEFT_OPPONENT: 1, PlayerSeat.RIGHT_OPPONENT: 17},
        state_matrix=tuple(1 for _ in range(54)),
    )

    recommendations = service.recommend(view)

    assert recommendations[0].action_label == "火箭 小王,大王"
