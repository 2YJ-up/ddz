from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelConfig:
    card_detector_path: Path
    policy_model_path: Path


@dataclass(frozen=True, slots=True)
class CvConfig:
    confidence_threshold: float
    nms_iou_threshold: float


@dataclass(frozen=True, slots=True)
class RlConfig:
    top_n: int


@dataclass(frozen=True, slots=True)
class UiConfig:
    overlay_offset_x: int
    overlay_offset_y: int
    overlay_width: int
    overlay_height: int


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    frame_budget_ms: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    window_title: str
    capture_fps: int
    models: ModelConfig
    cv: CvConfig
    rl: RlConfig
    ui: UiConfig
    pipeline: PipelineConfig


def _read_mapping(config_path: Path) -> Mapping[str, Any]:
    raw_text = config_path.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text)
    return raw_data


def load_config(config_path: str | Path = "config/default.json") -> AppConfig:
    path = Path(config_path)
    raw_data = _read_mapping(path)
    model_data = raw_data["models"]
    cv_data = raw_data["cv"]
    rl_data = raw_data["rl"]
    ui_data = raw_data["ui"]
    pipeline_data = raw_data["pipeline"]

    config = AppConfig(
        window_title=str(raw_data["window_title"]),
        capture_fps=int(raw_data["capture_fps"]),
        models=ModelConfig(
            card_detector_path=Path(model_data["card_detector_path"]),
            policy_model_path=Path(model_data["policy_model_path"]),
        ),
        cv=CvConfig(
            confidence_threshold=float(cv_data["confidence_threshold"]),
            nms_iou_threshold=float(cv_data["nms_iou_threshold"]),
        ),
        rl=RlConfig(top_n=int(rl_data["top_n"])),
        ui=UiConfig(
            overlay_offset_x=int(ui_data["overlay_offset_x"]),
            overlay_offset_y=int(ui_data["overlay_offset_y"]),
            overlay_width=int(ui_data["overlay_width"]),
            overlay_height=int(ui_data["overlay_height"]),
        ),
        pipeline=PipelineConfig(frame_budget_ms=int(pipeline_data["frame_budget_ms"])),
    )
    return config
