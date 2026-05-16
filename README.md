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
