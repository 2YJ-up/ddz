from pathlib import Path

from src.cv.inference import CardDetector, CardDetectorConfig
from src.domain.actions import FrameBuffer, WindowRect


def test_detector_returns_empty_when_model_missing() -> None:
    detector = CardDetector(
        CardDetectorConfig(
            model_path=Path("models/missing.onnx"),
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
        )
    )
    detector.load()
    frame = FrameBuffer(
        pixels=b"\x00\x00\x00",
        width=1,
        height=1,
        timestamp_ms=1,
        window_rect=WindowRect(left=0, top=0, width=1, height=1),
        frame_id="f",
    )

    detections = detector.detect(frame)

    assert detections == ()
