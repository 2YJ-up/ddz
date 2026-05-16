import asyncio

from src.app import build_runtime
from src.config import load_config


def test_loads_default_config_and_runs_missing_window_once() -> None:
    config = load_config("config/default.json")
    runtime = build_runtime(config)

    try:
        snapshot = asyncio.run(runtime.run_once())
    finally:
        runtime.close()

    assert snapshot.frame.width == 0
    assert tuple(snapshot.elapsed_ms_by_step.keys()) == (
        "IO_READ",
        "CV_INFERENCE",
        "STATE_NORMALIZATION",
        "RL_FORWARD",
        "UI_RENDER",
    )
