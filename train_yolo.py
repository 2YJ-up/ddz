from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_YAML = Path(
    r"C:\Users\jie\Downloads\DouDiZhu V2.v9-roboflow-instant-4--eval-.yolov8\data.yaml"
)
DEFAULT_OUTPUT_MODEL = Path("models/yolov8_cards.pt")


@dataclass(frozen=True, slots=True)
class YoloTrainingConfig:
    data_yaml: Path
    pretrained_weight: str
    epochs: int
    image_size: int
    output_model: Path
    project_dir: Path
    run_name: str


class YoloTrainer:
    def __init__(self, config: YoloTrainingConfig) -> None:
        self.config = config
        self._configure_runtime()

    def train(self) -> Path:
        from ultralytics import YOLO

        self._validate_inputs()
        self.config.output_model.parent.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.config.pretrained_weight)
        model.train(
            data=str(self.config.data_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            project=str(self.config.project_dir),
            name=self.config.run_name,
            exist_ok=True,
        )

        best_model_path = self._resolve_best_model_path()
        shutil.copy2(best_model_path, self.config.output_model)
        return self.config.output_model

    def _configure_runtime(self) -> None:
        runtime_dir = Path(".runtime").resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))

    def _validate_inputs(self) -> None:
        if not self.config.data_yaml.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {self.config.data_yaml}")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.config.image_size <= 0:
            raise ValueError("image_size must be greater than 0")

    def _resolve_best_model_path(self) -> Path:
        expected_path = self.config.project_dir / self.config.run_name / "weights" / "best.pt"
        best_model_path = expected_path
        if not best_model_path.exists():
            candidates = tuple(self.config.project_dir.rglob("best.pt"))
            if len(candidates) > 0:
                best_model_path = max(candidates, key=lambda item: item.stat().st_mtime)
        if not best_model_path.exists():
            raise FileNotFoundError(f"Training completed but best.pt was not found under: {self.config.project_dir}")
        return best_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n card detector and export best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="Path to Roboflow YOLOv8 data.yaml.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epoch count.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MODEL, help="Output model path.")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="doudizhu_cards", help="Ultralytics run name.")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = YoloTrainingConfig(
        data_yaml=args.data,
        pretrained_weight="yolov8n.pt",
        epochs=args.epochs,
        image_size=args.imgsz,
        output_model=args.output,
        project_dir=args.project,
        run_name=args.name,
    )
    trainer = YoloTrainer(config)
    output_model = trainer.train()
    print(f"Training complete. Exported model: {output_model}")


if __name__ == "__main__":
    main()
