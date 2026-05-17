import json
from argparse import Namespace
from pathlib import Path

import pytest

from app.live_state_writer import (
    LiveActionMemory,
    _cards_from_detections,
    _is_legal_table_shape,
    _load_table_override,
    _recommendation_readiness,
    _repair_visible_card_counts,
    _resolve_last_table_action,
    _select_table_cluster,
)
from app.overlay_client import RecommendationOverlay, _waiting_response
from src.adapters.perfectdou_adapter import PerfectDouAdapter
from src.adapters.vision_adapter import parse_recognized_frame_state
from src.core.belief_sampler import build_unknown_pool, sample_hidden_hands
from src.core.cards import CARD_LIMITS, count_cards
from src.core.visible_state import VisibleAction, VisibleGameState
from src.domain.actions import BoundingBox, CardDetection, PlayerSeat
from src.domain.cards import CardRank
from src.services.recommendation_service import RecommendationOptions, RecommendationService
from app.main import create_app


def _load_example(name: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "examples" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_recommendation_contract_returns_top_three_score_mode() -> None:
    service = RecommendationService()
    response = service.recommend(
        _load_example("midgame_state.json"),
        RecommendationOptions(n_samples=32, rollout_per_action=4, mode="perfectdou_monte_carlo", seed=1),
    )

    assert response["best_action"] == []
    assert len(response["top_actions"]) == 3
    assert response["top_actions"][0]["win_rate"] is None
    assert response["top_actions"][0]["score"] > 0
    assert response["belief_summary"]["samples_used"] == 32
    assert response["model_source"] == "douzero_adp"
    assert "win_rate 为 null" in " ".join(response["state_warnings"])


def test_v3_dry_run_keeps_low_confidence_warning() -> None:
    service = RecommendationService()
    response = service.dry_run(_load_example("recognized_state.json"))

    assert response["state_errors"] == []
    assert response["visible_state"]["recognition_confidence"] == 0.82
    assert "recognizer reported a low-confidence frame" in response["state_warnings"]


def test_v3_sampler_respects_counts_and_known_cards() -> None:
    recognized = parse_recognized_frame_state(_load_example("midgame_state.json"))
    state = recognized.to_visible_state()
    samples = sample_hidden_hands(state, n_samples=8, seed=2)
    public_cards = []
    for action in state.public_action_history:
        public_cards.extend(action.cards)

    assert len(build_unknown_pool(state)) == 30
    assert len(samples) == 8
    for sample in samples:
        assert len(sample.hands["landlord"]) == state.num_cards_left["landlord"]
        assert len(sample.hands["landlord_down"]) == state.num_cards_left["landlord_down"]
        assert "A" in sample.hands["landlord"]
        assert "2" in sample.hands["landlord"]
        assert "D" in sample.hands["landlord"]
        combined_cards = list(state.self_hand) + public_cards
        for player_cards in sample.hands.values():
            combined_cards.extend(player_cards)
        combined_counts = count_cards(combined_cards)
        for card, amount in combined_counts.items():
            assert amount <= CARD_LIMITS[card]


def test_v3_fastapi_recommend_route_endpoint() -> None:
    api = create_app()
    route = next(route for route in api.routes if getattr(route, "path", "") == "/recommend")
    response = route.endpoint({"state": _load_example("midgame_state.json"), "options": {"n_samples": 8}})

    assert response["best_action"] == []
    assert response["belief_summary"]["samples_used"] == 8
    assert response["model_source"] == "douzero_adp"


def test_v3_perfectdou_official_rules_are_used_when_repo_exists() -> None:
    repo = Path(__file__).resolve().parents[1] / "external" / "PerfectDou"
    if not repo.exists():
        pytest.skip("external/PerfectDou is not checked out")
    adapter = PerfectDouAdapter(repo_path=str(repo))
    state = parse_recognized_frame_state(_load_example("midgame_state.json")).to_visible_state()

    legal_actions = adapter.generate_legal_actions(state)

    assert adapter.available
    assert adapter.rules_available
    assert ["8", "8"] in legal_actions
    assert [] in legal_actions


def test_v3_visible_state_resets_current_trick_after_two_passes() -> None:
    state = VisibleGameState(
        self_position="landlord",
        acting_player="landlord",
        self_hand=["3", "4", "5", "6", "7"],
        public_action_history=[
            VisibleAction(player="landlord_down", cards=["A"], is_pass=False),
            VisibleAction(player="landlord_up", cards=[], is_pass=True),
            VisibleAction(player="landlord", cards=[], is_pass=True),
        ],
        num_cards_left={"landlord": 5, "landlord_down": 16, "landlord_up": 17},
        landlord_public_cards=[],
        recognition_confidence=0.99,
    )
    service = RecommendationService()
    response = service.recommend(state, RecommendationOptions(n_samples=0, mode="heuristic"))

    assert state.last_non_pass_action is None
    assert response["top_actions"][0]["action"] == ["3", "4", "5", "6", "7"]


def test_live_writer_marks_low_confidence_frame_not_ready() -> None:
    args = Namespace(
        min_recommend_confidence=0.45,
        min_many_cards_confidence=0.28,
        many_cards_threshold=8,
        acting_player="landlord",
        self_position="landlord",
    )

    ready, reason = _recommendation_readiness(args, ["6", "6", "9"], 0.14)

    assert not ready
    assert "暂不推荐" in reason


def test_live_writer_rejects_invalid_detected_table_shape() -> None:
    args = Namespace(
        min_recommend_confidence=0.45,
        min_many_cards_confidence=0.28,
        many_cards_threshold=8,
        min_table_confidence=0.45,
        acting_player="landlord",
        self_position="landlord",
    )

    ready, reason = _recommendation_readiness(args, ["K", "K"], 0.9, ["A", "X"], 0.9, False)

    assert not ready
    assert not _is_legal_table_shape(["A", "X"])
    assert "不是合法斗地主牌型" in reason


def test_live_writer_allows_many_visible_cards_with_lower_mean_confidence() -> None:
    args = Namespace(
        min_recommend_confidence=0.45,
        min_many_cards_confidence=0.28,
        many_cards_threshold=8,
        min_table_confidence=0.15,
        acting_player="landlord_down",
        self_position="landlord_down",
    )

    ready, reason = _recommendation_readiness(args, ["3", "4", "5", "6", "7", "8", "9", "10"], 0.32, ["7"], 0.17, True)

    assert ready
    assert reason == "可推荐"


def test_live_writer_selects_legal_table_cluster_for_response() -> None:
    args = Namespace(
        table_candidate_confidence=0.05,
        table_min_x_ratio=0.28,
        table_max_x_ratio=0.74,
        table_min_y_ratio=0.20,
        table_max_y_ratio=0.55,
        table_row_tolerance_ratio=0.08,
        table_gap_ratio=0.08,
    )
    frame = Namespace(width=2000, height=1000)
    detections = (
        _det(CardRank.FOUR, 0.21, 820, 280),
        _det(CardRank.FOUR, 0.19, 890, 280),
        _det(CardRank.FOUR, 0.18, 960, 280),
        _det(CardRank.THREE, 0.20, 1030, 280),
        _det(CardRank.JACK, 0.50, 1520, 80),
    )

    selected = _select_table_cluster(args, detections, frame)

    assert _cards_from_detections(selected) == ["3", "4", "4", "4"]


def test_v3_response_to_triple_with_pair_never_recommends_pair() -> None:
    service = RecommendationService()
    response = service.recommend(
        {
            "self_hand": ["K", "Q", "Q", "Q", "Q", "6"],
            "last_table_action": {"player": "landlord", "cards": ["3", "3", "3", "10", "10"], "is_pass": False},
            "public_action_history": [
                {"player": "landlord", "cards": ["3", "3", "3", "10", "10"], "is_pass": False}
            ],
            "player_position": "landlord_down",
            "acting_player": "landlord_down",
            "num_cards_left": {"landlord": 7, "landlord_down": 6, "landlord_up": 4},
            "landlord_public_cards": [],
            "recognition_confidence": 0.9,
        },
        RecommendationOptions(n_samples=0, mode="heuristic"),
    )

    actions = [item["action"] for item in response["top_actions"]]

    assert ["Q", "Q"] not in actions
    assert sorted(actions, key=len) == [[], ["Q", "Q", "Q", "Q"]]


def test_live_writer_keeps_previous_non_pass_when_intervening_player_passes() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=True,
        last_table_player="landlord",
        new_action_min_confidence=0.25,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=12,
    )
    memory = LiveActionMemory(last_non_pass_action={"player": "landlord", "cards": ["Q"], "is_pass": False})
    warnings: list[str] = []

    action, history, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord_up",
        candidate_cards=["2"],
        candidate_confidence=0.9,
        manual=False,
        warnings=warnings,
    )

    assert action == {"player": "landlord", "cards": ["Q"], "is_pass": False}
    assert history == [action]
    assert source == "remembered_after_pass"
    assert "沿用最近一次非过牌" in warnings[0]


def test_live_writer_accepts_new_action_from_expected_previous_player() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=True,
        last_table_player="landlord",
        new_action_min_confidence=0.25,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=12,
    )
    memory = LiveActionMemory(last_non_pass_action={"player": "landlord_up", "cards": ["Q"], "is_pass": False})
    warnings: list[str] = []

    action, _, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord",
        candidate_cards=["2"],
        candidate_confidence=0.9,
        manual=False,
        warnings=warnings,
    )

    assert action == {"player": "landlord", "cards": ["2"], "is_pass": False}
    assert source == "detected"


def test_live_writer_refreshes_same_player_new_shape_instead_of_sticking_to_old_single() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=True,
        last_table_player="landlord",
        new_action_min_confidence=0.15,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=12,
    )
    memory = LiveActionMemory(last_non_pass_action={"player": "landlord", "cards": ["4"], "is_pass": False})

    action, _, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord",
        candidate_cards=["2", "2"],
        candidate_confidence=0.16,
        manual=False,
        warnings=[],
    )

    assert action == {"player": "landlord", "cards": ["2", "2"], "is_pass": False}
    assert source == "detected"


def test_live_writer_can_accept_new_action_from_non_default_player_when_not_strict() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=False,
        last_table_player="landlord_down",
        new_action_min_confidence=0.15,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=12,
    )
    memory = LiveActionMemory(last_non_pass_action={"player": "landlord_down", "cards": ["4"], "is_pass": False})

    action, _, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord",
        candidate_cards=["J"],
        candidate_confidence=0.35,
        manual=False,
        warnings=[],
    )

    assert action == {"player": "landlord", "cards": ["J"], "is_pass": False}
    assert source == "detected"


def test_live_writer_expires_stale_memory_without_new_cards() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=True,
        last_table_player="landlord",
        new_action_min_confidence=0.15,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=2,
    )
    memory = LiveActionMemory(
        last_non_pass_action={"player": "landlord", "cards": ["4"], "is_pass": False},
        retained_frames=2,
    )
    warnings: list[str] = []

    action, history, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord",
        candidate_cards=[],
        candidate_confidence=0.0,
        manual=False,
        warnings=warnings,
    )

    assert action is None
    assert history == []
    assert source == "expired"
    assert memory.last_non_pass_action is None


def test_live_writer_force_lead_override_clears_previous_action_memory() -> None:
    args = Namespace(
        remember_last_action=True,
        prefer_last_table_player=True,
        last_table_player="landlord",
        new_action_min_confidence=0.15,
        same_player_refresh_after_frames=1,
        max_memory_retained_frames=12,
    )
    memory = LiveActionMemory(
        last_non_pass_action={"player": "landlord", "cards": ["2"], "is_pass": False},
        retained_frames=3,
    )

    action, history, source = _resolve_last_table_action(
        args=args,
        memory=memory,
        candidate_player="landlord",
        candidate_cards=[],
        candidate_confidence=0.0,
        manual=False,
        warnings=[],
        force_lead=True,
    )

    assert action is None
    assert history == []
    assert source == "forced_lead"
    assert memory.last_non_pass_action is None
    assert memory.retained_frames == 0


def test_live_override_file_can_force_free_lead(tmp_path: Path) -> None:
    override_file = tmp_path / "live_override.json"
    override_file.write_text(
        json.dumps({"enabled": True, "force_lead": True, "updated_at_ms": 0, "ttl_seconds": 30}),
        encoding="utf-8",
    )
    args = Namespace(override_file=override_file, last_table_player="landlord_up", override_ttl_seconds=30.0)

    player, cards, force_lead, warning = _load_table_override(args)

    assert player == "landlord_up"
    assert cards == []
    assert force_lead is True
    assert "自由出牌" in warning


def test_live_writer_repairs_duplicate_small_joker_as_big_joker() -> None:
    self_hand, table_cards, warnings = _repair_visible_card_counts(["X", "K"], ["X"])

    assert self_hand == ["K", "D"]
    assert table_cards == ["X"]
    assert "大小王" in warnings[0]


def test_overlay_waiting_state_is_not_displayed_as_pass() -> None:
    response = _waiting_response(
        {"frame_id": "live:1", "recognition_confidence": 0.14, "self_hand": ["6", "6", "9"]},
        "识别置信度过低",
        [],
    )

    text = RecommendationOverlay._format_response(object(), response)

    assert "当前推荐：等待识别" in text
    assert "当前推荐：过" not in text


def _det(rank: CardRank, confidence: float, x: int, y: int) -> CardDetection:
    return CardDetection(
        card_rank=rank,
        confidence=confidence,
        bbox=BoundingBox(x=x, y=y, width=80, height=120),
        seat_region=PlayerSeat.TABLE,
    )
