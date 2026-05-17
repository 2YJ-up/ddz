# Doudizhu Master Assistant

This repository contains the first backend slice for a Doudizhu AI assistant
using a PerfectDou-style decision pipeline:

`IO_READ -> CV_INFERENCE -> STATE_NORMALIZATION -> RL_FORWARD -> UI_RENDER`

The current implementation focuses on stable contracts, card/action domain
types, rule validation, normalized state replay, and a single-thread pipeline
orchestrator. Model and UI integrations are intentionally isolated behind
module boundaries.

## Layout

```text
config/default.json
src/config.py
src/domain/cards.py
src/domain/actions.py
src/core/rule_engine.py
src/core/state_manager.py
src/io/screen_capture.py
src/cv/inference.py
src/ai/rl_forward.py
src/ui/overlay_render.py
src/main_pipeline.py
tests/
```

Model weights are expected at:

```text
models/yolov8_cards.pt
models/perfectdou_actor.onnx
```

The repository does not include those binary assets.

## Test

```powershell
python -m pytest
```

## Run

```powershell
python run_assistant.py
```

The default command runs the continuous assistant loop and prints one readable
status line per second. Press `Ctrl+C` to stop it.

Useful diagnostics:

```powershell
python run_assistant.py --health
python run_assistant.py --debug-windows
python run_assistant.py --once
python run_assistant.py --bring-window-front
python run_assistant.py --save-frame debug_capture.png
python run_assistant.py --save-detections debug_detect.png
```

The screen capturer uses visible desktop pixels from the MuMu client area, so
keep the game window visible and uncovered while the assistant is running.
The overlay is topmost and is automatically clamped back onto the visible
screen if the game window is too wide for an outside panel.

## Model Assets

Place model files under `models/`:

```text
models/yolov8_cards.pt
models/perfectdou_actor.onnx
```

`CardDetector` supports Ultralytics `.pt` models and OpenCV DNN `.onnx` card
detectors. `RlForwardService` supports ONNX Runtime for the policy model and
falls back to legal-action heuristic scoring when the model file is absent.

The pipeline keeps module ownership strict:

- `IO_READ`: window capture only.
- `CV_INFERENCE`: frame-to-card detections only.
- `STATE_NORMALIZATION`: base truths and derived state views.
- `RL_FORWARD`: legal action scoring and recommendation.
- `UI_RENDER`: overlay frame/window rendering only.

## v3 Visible-Information API

The v3 product-facing path accepts a recognized visible state JSON and returns
Top 3 recommendations without reading hidden cards or controlling the game
client.

```powershell
python -m app.main --input examples/recognized_state.json --dry-run --pretty
python -m app.main --input examples/midgame_state.json --mode perfectdou_monte_carlo --pretty
python -m app.main --input examples/midgame_state.json --mode douzero_adp --pretty
python -m app.main --serve --host 127.0.0.1 --port 8000
python -m app.live_state_writer --output examples/live_test.json --self-position landlord --acting-player landlord
python -m app.overlay_client --input examples/live_test.json
```

`POST /recommend` accepts:

```json
{
  "state": {"self_hand": ["3", "3"]},
  "options": {"n_samples": 256, "rollout_per_action": 64, "mode": "perfectdou_monte_carlo"}
}
```

When PerfectDou rollout is unavailable, the response uses `score` for ranking
and leaves `win_rate` as `null`.

The API server does not draw anything by itself. Use `app.overlay_client` in a
second terminal to show the latest recommendation from a JSON state file in a
topmost overlay. Press `Esc` to close it and drag the panel with the mouse to
move it.

For live capture, run `app.live_state_writer` in a third terminal. It writes the
recognized hand into `examples/live_test.json`. If the game is in bidding,
doubling, settlement, or the detector sees no cards, the overlay will show a
waiting message instead of repeating stale recommendations.

`external/PerfectDou` is treated as a read-only third-party checkout. In the
current Windows/Python runtime the official pure-Python rule modules can be
used, while the official policy encoder is packaged as a Linux Python 3.7
shared object, so full PerfectDou rollout still requires a compatible runtime.
The project also uses the DouZero ADP checkpoints bundled in that checkout as a
usable strong-model fallback on the current machine.

```powershell
git clone --depth 1 https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou.git external/PerfectDou
```
