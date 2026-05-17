from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.domain.actions import BoundingBox, CardDetection, FrameBuffer, PlayerSeat
from src.domain.cards import rank_from_label


@dataclass(frozen=True, slots=True)
class CardDetectorConfig:
    model_path: Path
    confidence_threshold: float
    nms_iou_threshold: float
    input_size: int = 640
    class_names: Mapping[int, str] = field(default_factory=dict)


class CardDetector:
    def __init__(self, config: CardDetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._cv2: Any = None
        self._backend = "none"
        self._model_loaded = False
        self._class_names = self._resolve_class_names(config.class_names)

    def load(self) -> None:
        self._model = None
        self._cv2 = None
        self._backend = "none"
        self._model_loaded = False

        if self.config.model_path.exists():
            suffix = self.config.model_path.suffix.lower()
            if suffix == ".pt":
                self._load_ultralytics()
            elif suffix == ".onnx":
                self._load_cv2_onnx()

    def is_loaded(self) -> bool:
        result = self._model_loaded
        return result

    def backend_name(self) -> str:
        result = self._backend
        return result

    def class_names(self) -> Mapping[int, str]:
        result = dict(self._class_names)
        return result

    def detect(self, frame: FrameBuffer) -> tuple[CardDetection, ...]:
        detections: tuple[CardDetection, ...] = ()
        can_detect = self._model_loaded and frame.width > 0 and frame.height > 0
        if can_detect:
            image = self._frame_to_bgr_array(frame)
            if image.size > 0 and self._backend == "ultralytics":
                detections = self._detect_with_ultralytics(image, frame)
            elif image.size > 0 and self._backend == "cv2_onnx":
                detections = self._detect_with_cv2_onnx(image, frame)
        return detections

    def _load_ultralytics(self) -> None:
        try:
            self._ensure_ultralytics_config_dir()
            module = importlib.import_module("ultralytics")
            yolo_class = getattr(module, "YOLO")
            self._model = yolo_class(str(self.config.model_path))
            self._backend = "ultralytics"
            self._model_loaded = True
            self._merge_model_names(getattr(self._model, "names", {}))
        except (ImportError, AttributeError, OSError, RuntimeError):
            self._model = None
            self._backend = "none"
            self._model_loaded = False

    def _ensure_ultralytics_config_dir(self) -> None:
        runtime_dir = Path(".runtime").resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))

    def _load_cv2_onnx(self) -> None:
        try:
            cv2_module = importlib.import_module("cv2")
            self._cv2 = cv2_module
            self._model = self._cv2.dnn.readNetFromONNX(str(self.config.model_path))
            self._backend = "cv2_onnx"
            self._model_loaded = True
        except (ImportError, AttributeError, OSError, RuntimeError):
            self._cv2 = None
            self._model = None
            self._backend = "none"
            self._model_loaded = False

    def _frame_to_bgr_array(self, frame: FrameBuffer) -> np.ndarray:
        expected_length = frame.width * frame.height * 3
        image = np.empty((0, 0, 3), dtype=np.uint8)
        if len(frame.pixels) == expected_length:
            image = np.frombuffer(frame.pixels, dtype=np.uint8).reshape((frame.height, frame.width, 3))
        return image

    def _detect_with_ultralytics(self, image: np.ndarray, frame: FrameBuffer) -> tuple[CardDetection, ...]:
        detections: list[CardDetection] = []
        predictions = self._model.predict(
            source=image,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_iou_threshold,
            max_det=80,
            verbose=False,
        )
        if len(predictions) > 0:
            result = predictions[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                index = 0
                while index < len(classes):
                    detection = self._build_detection_from_xyxy(
                        class_index=int(classes[index]),
                        confidence=float(confidences[index]),
                        xyxy=xyxy[index],
                        frame=frame,
                    )
                    if detection is not None:
                        detections.append(detection)
                    index += 1
        ordered = tuple(sorted(detections, key=lambda item: item.confidence, reverse=True))
        return ordered

    def _detect_with_cv2_onnx(self, image: np.ndarray, frame: FrameBuffer) -> tuple[CardDetection, ...]:
        blob = self._cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(self.config.input_size, self.config.input_size),
            swapRB=True,
            crop=False,
        )
        self._model.setInput(blob)
        raw_outputs = self._model.forward()
        candidates = self._decode_yolo_outputs(raw_outputs, frame)
        detections = self._apply_nms(candidates, frame)
        return detections

    def _decode_yolo_outputs(
        self,
        raw_outputs: np.ndarray,
        frame: FrameBuffer,
    ) -> tuple[tuple[int, float, tuple[float, float, float, float]], ...]:
        output = np.squeeze(raw_outputs)
        if output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T

        scale_x = float(frame.width) / float(self.config.input_size)
        scale_y = float(frame.height) / float(self.config.input_size)
        candidates: list[tuple[int, float, tuple[float, float, float, float]]] = []
        row_index = 0
        while row_index < output.shape[0]:
            row = output[row_index]
            if row.shape[0] >= 6:
                class_scores = row[4:]
                class_index = int(np.argmax(class_scores))
                confidence = float(class_scores[class_index])
                if confidence >= self.config.confidence_threshold:
                    center_x = float(row[0]) * scale_x
                    center_y = float(row[1]) * scale_y
                    width = float(row[2]) * scale_x
                    height = float(row[3]) * scale_y
                    xyxy = (
                        center_x - width / 2.0,
                        center_y - height / 2.0,
                        center_x + width / 2.0,
                        center_y + height / 2.0,
                    )
                    candidates.append((class_index, confidence, xyxy))
            row_index += 1
        result = tuple(candidates)
        return result

    def _apply_nms(
        self,
        candidates: tuple[tuple[int, float, tuple[float, float, float, float]], ...],
        frame: FrameBuffer,
    ) -> tuple[CardDetection, ...]:
        detections: tuple[CardDetection, ...] = ()
        if len(candidates) > 0:
            boxes = []
            confidences = []
            for _, confidence, xyxy in candidates:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])])
                confidences.append(confidence)
            indexes = self._cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.config.confidence_threshold,
                self.config.nms_iou_threshold,
            )
            flat_indexes = np.array(indexes).reshape(-1) if len(indexes) > 0 else np.array([], dtype=np.int32)
            selected: list[CardDetection] = []
            item_index = 0
            while item_index < len(flat_indexes):
                candidate_index = int(flat_indexes[item_index])
                class_index, confidence, xyxy = candidates[candidate_index]
                detection = self._build_detection_from_xyxy(class_index, confidence, np.array(xyxy), frame)
                if detection is not None:
                    selected.append(detection)
                item_index += 1
            detections = tuple(sorted(selected, key=lambda item: item.confidence, reverse=True))
        return detections

    def _build_detection_from_xyxy(
        self,
        class_index: int,
        confidence: float,
        xyxy: np.ndarray | tuple[float, float, float, float],
        frame: FrameBuffer,
    ) -> CardDetection | None:
        label = self._class_names.get(class_index, "")
        rank = rank_from_label(label)
        detection: CardDetection | None = None
        if rank is not None:
            x1 = max(0, int(float(xyxy[0])))
            y1 = max(0, int(float(xyxy[1])))
            x2 = min(frame.width, max(x1, int(float(xyxy[2]))))
            y2 = min(frame.height, max(y1, int(float(xyxy[3]))))
            bbox = BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))
            if bbox.width > 0 and bbox.height > 0:
                seat_region = self._infer_seat_region(bbox, frame)
                detection = CardDetection(
                    card_rank=rank,
                    confidence=confidence,
                    bbox=bbox,
                    seat_region=seat_region,
                )
        return detection

    def _infer_seat_region(self, bbox: BoundingBox, frame: FrameBuffer) -> PlayerSeat:
        center_x = bbox.x + bbox.width / 2.0
        center_y = bbox.y + bbox.height / 2.0
        seat = PlayerSeat.SELF
        if frame.height > 0 and center_y < frame.height * 0.55:
            is_table_card = (
                frame.width > 0
                and center_y >= frame.height * 0.18
                and center_y <= frame.height * 0.68
                and center_x >= frame.width * 0.25
                and center_x <= frame.width * 0.75
            )
            if is_table_card:
                seat = PlayerSeat.TABLE
            elif frame.width > 0 and center_x < frame.width / 2.0:
                seat = PlayerSeat.LEFT_OPPONENT
            else:
                seat = PlayerSeat.RIGHT_OPPONENT
        return seat

    def _resolve_class_names(self, class_names: Mapping[int, str]) -> dict[int, str]:
        resolved = dict(self._default_class_names())
        for class_index, label in class_names.items():
            resolved[int(class_index)] = str(label)
        return resolved

    def _merge_model_names(self, class_names: Mapping[int, str]) -> None:
        for class_index, label in class_names.items():
            normalized_index = int(class_index)
            if normalized_index not in self._class_names:
                self._class_names[normalized_index] = str(label)

    def _default_class_names(self) -> dict[int, str]:
        names = {
            0: "small_joker",
            1: "A",
            2: "10",
            3: "J",
            4: "Q",
            5: "K",
            6: "2",
            7: "3",
            8: "4",
            9: "5",
            10: "6",
            11: "7",
            12: "8",
            13: "9",
        }
        return names
