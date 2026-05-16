from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.domain.actions import CardDetection, FrameBuffer


@dataclass(frozen=True, slots=True)
class CardDetectorConfig:
    model_path: Path
    confidence_threshold: float
    nms_iou_threshold: float


class CardDetector:
    def __init__(self, config: CardDetectorConfig) -> None:
        self.config = config
        self._model_loaded = False

    def load(self) -> None:
        self._model_loaded = self.config.model_path.exists()

    def is_loaded(self) -> bool:
        result = self._model_loaded
        return result

    def detect(self, frame: FrameBuffer) -> tuple[CardDetection, ...]:
        detections: tuple[CardDetection, ...] = ()
        return detections
