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

## Run One Pipeline Frame

```powershell
python run_assistant.py
```

The default configuration keeps the overlay window disabled and safely returns
an empty frame when the game window is not found. To enable the topmost overlay,
set `ui.enable_window` to `true` in `config/default.json`.

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
