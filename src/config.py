from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.domain.actions import PlayerSeat


@dataclass(frozen=True, slots=True)
class ModelConfig:
    card_detector_path: Path
    policy_model_path: Path


@dataclass(frozen=True, slots=True)
class CvConfig:
    confidence_threshold: float
    nms_iou_threshold: float
    input_size: int
    class_names: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class RlConfig:
    top_n: int
    input_name: str
    output_name: str


@dataclass(frozen=True, slots=True)
class StateConfig:
    landlord_seat: PlayerSeat
    first_turn: PlayerSeat
    bootstrap_from_self_detections: bool
    min_self_cards_to_start: int
    bootstrap_stable_frames: int


@dataclass(frozen=True, slots=True)
class UiConfig:
    overlay_offset_x: int
    overlay_offset_y: int
    overlay_width: int
    overlay_height: int
    enable_window: bool
    place_outside_capture: bool


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
    state: StateConfig
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
    state_data = raw_data["state"]
    ui_data = raw_data["ui"]
    pipeline_data = raw_data["pipeline"]
    class_names = {int(key): str(value) for key, value in cv_data.get("class_names", {}).items()}

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
            input_size=int(cv_data.get("input_size", 640)),
            class_names=class_names,
        ),
        rl=RlConfig(
            top_n=int(rl_data["top_n"]),
            input_name=str(rl_data.get("input_name", "")),
            output_name=str(rl_data.get("output_name", "")),
        ),
        state=StateConfig(
            landlord_seat=PlayerSeat(str(state_data["landlord_seat"])),
            first_turn=PlayerSeat(str(state_data["first_turn"])),
            bootstrap_from_self_detections=bool(state_data["bootstrap_from_self_detections"]),
            min_self_cards_to_start=int(state_data.get("min_self_cards_to_start", 1)),
            bootstrap_stable_frames=int(state_data.get("bootstrap_stable_frames", 1)),
        ),
        ui=UiConfig(
            overlay_offset_x=int(ui_data["overlay_offset_x"]),
            overlay_offset_y=int(ui_data["overlay_offset_y"]),
            overlay_width=int(ui_data["overlay_width"]),
            overlay_height=int(ui_data["overlay_height"]),
            enable_window=bool(ui_data.get("enable_window", False)),
            place_outside_capture=bool(ui_data.get("place_outside_capture", True)),
        ),
        pipeline=PipelineConfig(frame_budget_ms=int(pipeline_data["frame_budget_ms"])),
    )
    return config
