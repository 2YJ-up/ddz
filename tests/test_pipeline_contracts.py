import asyncio
import io
import pathlib
import tokenize

from src.domain.actions import (
    ActionRecommendation,
    CardAction,
    GameStateView,
    PlayerSeat,
    RenderFrame,
    WindowRect,
)
from src.domain.cards import CardRank
from src.main_pipeline import MainPipeline


class FakeScreenCapture:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def capture(self):
        from src.domain.actions import FrameBuffer

        self.calls.append("IO_READ")
        frame = FrameBuffer(
            pixels=b"",
            width=1,
            height=1,
            timestamp_ms=1,
            window_rect=WindowRect(left=10, top=20, width=300, height=200),
            frame_id="f1",
        )
        return frame


class FakeCardDetector:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def detect(self, frame):
        self.calls.append("CV_INFERENCE")
        detections = ()
        return detections


class FakeStateNormalizer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def normalize(self, detections):
        self.calls.append("STATE_NORMALIZATION")
        view = GameStateView(
            current_turn=PlayerSeat.SELF,
            landlord_seat=PlayerSeat.SELF,
            action_log=(),
            current_trick_action=None,
            public_played_counts={},
            known_unseen_counts={},
            hand_counts={PlayerSeat.SELF: 0, PlayerSeat.LEFT_OPPONENT: 0, PlayerSeat.RIGHT_OPPONENT: 0},
            state_matrix=tuple(0 for _ in range(54)),
        )
        return view


class FakePolicyForward:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def recommend(self, state_view):
        self.calls.append("RL_FORWARD")
        recommendation = ActionRecommendation(
            action=CardAction(actor_seat=PlayerSeat.SELF, ranks=(CardRank.THREE,)),
            probability=1.0,
            expected_win_rate=0.5,
            reason_code="test",
        )
        return (recommendation,)


class FakeOverlayRenderer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def render(self, recommendations, game_window_rect):
        self.calls.append("UI_RENDER")
        frame = RenderFrame(
            text_blocks=("THREE",),
            highlight_ranks=(CardRank.THREE,),
            anchor_rect=game_window_rect,
        )
        return frame


def test_pipeline_runs_in_documented_order() -> None:
    calls: list[str] = []
    pipeline = MainPipeline(
        screen_capture=FakeScreenCapture(calls),
        card_detector=FakeCardDetector(calls),
        state_normalizer=FakeStateNormalizer(calls),
        policy_forward=FakePolicyForward(calls),
        overlay_renderer=FakeOverlayRenderer(calls),
        frame_budget_ms=1000,
    )

    snapshot = asyncio.run(pipeline.run_once())

    assert calls == ["IO_READ", "CV_INFERENCE", "STATE_NORMALIZATION", "RL_FORWARD", "UI_RENDER"]
    assert snapshot.render_frame.highlight_ranks == (CardRank.THREE,)
    assert tuple(snapshot.elapsed_ms_by_step.keys()) == tuple(calls)


def test_source_has_no_loop_escape_keywords() -> None:
    source_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    forbidden_tokens: list[tuple[str, str]] = []
    for path in source_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.NAME and token.string in {"break", "continue"}:
                forbidden_tokens.append((str(path), token.string))

    assert forbidden_tokens == []
