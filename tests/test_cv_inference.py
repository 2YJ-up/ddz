from pathlib import Path

from src.cv.inference import CardDetector, CardDetectorConfig
from src.domain.actions import BoundingBox, FrameBuffer, PlayerSeat, WindowRect
from src.domain.cards import CardRank, rank_from_label


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


def test_detector_config_class_names_take_precedence() -> None:
    detector = CardDetector(
        CardDetectorConfig(
            model_path=Path("models/missing.pt"),
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
            class_names={0: "3"},
        )
    )

    detector._merge_model_names({0: "0", 1: "4"})

    assert detector._class_names[0] == "3"
    assert detector._class_names[1] == "A"


def test_trained_dataset_numeric_labels_map_to_card_ranks() -> None:
    assert rank_from_label("0") is CardRank.SMALL_JOKER
    assert rank_from_label("1") is CardRank.ACE
    assert rank_from_label("13") is CardRank.KING


def test_detector_classifies_center_played_cards_as_table_region() -> None:
    detector = CardDetector(
        CardDetectorConfig(
            model_path=Path("models/missing.pt"),
            confidence_threshold=0.5,
            nms_iou_threshold=0.5,
        )
    )
    frame = FrameBuffer(
        pixels=b"",
        width=1000,
        height=800,
        timestamp_ms=1,
        window_rect=WindowRect(left=0, top=0, width=1000, height=800),
        frame_id="f",
    )

    seat = detector._infer_seat_region(BoundingBox(x=450, y=260, width=40, height=70), frame)

    assert seat is PlayerSeat.TABLE
