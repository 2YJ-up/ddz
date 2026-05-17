import json
from pathlib import Path

from src.adapters.vision_adapter import parse_recognized_frame_state
from src.core.belief_sampler import build_unknown_pool, sample_hidden_hands
from src.core.cards import CARD_LIMITS, count_cards
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

    assert response["best_action"] == ["8", "8"]
    assert len(response["top_actions"]) == 3
    assert response["top_actions"][0]["win_rate"] is None
    assert response["top_actions"][0]["score"] > 0
    assert response["belief_summary"]["samples_used"] == 32
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

    assert response["best_action"] == ["8", "8"]
    assert response["belief_summary"]["samples_used"] == 8
