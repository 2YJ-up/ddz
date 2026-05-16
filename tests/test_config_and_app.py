import asyncio
from dataclasses import replace

from src.app import build_runtime
from src.config import load_config
from src.domain.cards import CardRank
from src.domain.actions import WindowRect
from src.ui.overlay_render import OverlayConfig, OverlayRenderer


def test_loads_default_config_and_runs_missing_window_once() -> None:
    config = replace(load_config("config/default.json"), window_title="__missing_doudizhu_window__")
    runtime = build_runtime(config)

    try:
        snapshot = asyncio.run(runtime.run_once())
    finally:
        runtime.close()

    assert snapshot.frame.width == 0
    assert snapshot.state_view.hand_counts[config.state.landlord_seat] == 0
    assert tuple(snapshot.elapsed_ms_by_step.keys()) == (
        "IO_READ",
        "CV_INFERENCE",
        "STATE_NORMALIZATION",
        "RL_FORWARD",
        "UI_RENDER",
    )


def test_default_config_uses_stable_observation_bootstrap() -> None:
    config = load_config("config/default.json")

    assert config.cv.confidence_threshold == 0.35
    assert config.state.min_self_cards_to_start == 1
    assert config.state.bootstrap_stable_frames == 1


def test_overlay_anchor_stays_visible_when_window_fills_screen() -> None:
    renderer = OverlayRenderer(
        OverlayConfig(
            offset_x=20,
            offset_y=20,
            width=360,
            height=140,
            enable_window=False,
            place_outside_capture=True,
        )
    )
    try:
        screen_width, screen_height = renderer._screen_size()
        rect = renderer._build_anchor_rect(WindowRect(left=0, top=0, width=screen_width, height=screen_height))
    finally:
        renderer.close()

    assert 0 <= rect.left <= max(0, screen_width - rect.width)
    assert 0 <= rect.top <= max(0, screen_height - rect.height)


def test_overlay_uses_chinese_labels_for_pass_and_jokers() -> None:
    renderer = OverlayRenderer(
        OverlayConfig(
            offset_x=20,
            offset_y=20,
            width=360,
            height=140,
            enable_window=False,
            place_outside_capture=True,
        )
    )
    try:
        assert renderer._format_ranks(()) == "过"
        assert renderer._format_ranks((CardRank.SMALL_JOKER, CardRank.BIG_JOKER)) == "小王,大王"
    finally:
        renderer.close()
