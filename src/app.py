from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.ai.rl_forward import RlForwardConfig, RlForwardService
from src.config import AppConfig, load_config
from src.core.rule_engine import RuleEngine
from src.core.state_manager import StateManager
from src.core.state_normalizer import StateNormalizerConfig, StateObservationNormalizer
from src.cv.inference import CardDetector, CardDetectorConfig
from src.io.screen_capture import ScreenCapture, ScreenCaptureConfig
from src.main_pipeline import MainPipeline, PipelineSnapshot
from src.ui.overlay_render import OverlayConfig, OverlayRenderer


@dataclass(slots=True)
class AssistantRuntime:
    config: AppConfig
    pipeline: MainPipeline
    screen_capture: ScreenCapture
    card_detector: CardDetector
    rl_forward: RlForwardService
    overlay_renderer: OverlayRenderer

    async def run_once(self) -> PipelineSnapshot:
        snapshot = await self.pipeline.run_once()
        return snapshot

    async def run_frames(self, frame_count: int) -> tuple[PipelineSnapshot, ...]:
        snapshots: list[PipelineSnapshot] = []
        frame_interval_seconds = 1.0 / float(max(1, self.config.capture_fps))
        index = 0
        while index < frame_count:
            start = time.perf_counter()
            snapshot = await self.run_once()
            snapshots.append(snapshot)
            elapsed = time.perf_counter() - start
            remaining = frame_interval_seconds - elapsed
            if remaining > 0.0:
                await asyncio.sleep(remaining)
            index += 1
        result = tuple(snapshots)
        return result

    async def run_loop(self, status_interval_seconds: float) -> None:
        next_status_at = 0.0
        frame_interval_seconds = 1.0 / float(max(1, self.config.capture_fps))
        while True:
            start = time.perf_counter()
            snapshot = await self.run_once()
            now = time.perf_counter()
            if now >= next_status_at:
                print(format_snapshot_status(snapshot), flush=True)
                next_status_at = now + status_interval_seconds
            elapsed = time.perf_counter() - start
            remaining = frame_interval_seconds - elapsed
            if remaining > 0.0:
                await asyncio.sleep(remaining)

    def close(self) -> None:
        self.screen_capture.close()
        self.overlay_renderer.close()


def build_runtime(config: AppConfig) -> AssistantRuntime:
    rule_engine = RuleEngine()
    state_manager = StateManager(rule_engine=rule_engine)
    screen_capture = ScreenCapture(ScreenCaptureConfig(window_title=config.window_title))
    card_detector = CardDetector(
        CardDetectorConfig(
            model_path=config.models.card_detector_path,
            confidence_threshold=config.cv.confidence_threshold,
            nms_iou_threshold=config.cv.nms_iou_threshold,
            input_size=config.cv.input_size,
            class_names=config.cv.class_names,
        )
    )
    card_detector.load()
    state_normalizer = StateObservationNormalizer(
        config=StateNormalizerConfig(
            landlord_seat=config.state.landlord_seat,
            first_turn=config.state.first_turn,
            bootstrap_from_self_detections=config.state.bootstrap_from_self_detections,
            min_self_cards_to_start=config.state.min_self_cards_to_start,
            bootstrap_stable_frames=config.state.bootstrap_stable_frames,
            live_self_stable_frames=config.state.live_self_stable_frames,
        ),
        state_manager=state_manager,
    )
    rl_forward = RlForwardService(
        config=RlForwardConfig(
            model_path=config.models.policy_model_path,
            top_n=config.rl.top_n,
            input_name=config.rl.input_name,
            output_name=config.rl.output_name,
        ),
        rule_engine=rule_engine,
    )
    rl_forward.load()
    overlay_renderer = OverlayRenderer(
        OverlayConfig(
            offset_x=config.ui.overlay_offset_x,
            offset_y=config.ui.overlay_offset_y,
            width=config.ui.overlay_width,
            height=config.ui.overlay_height,
            enable_window=config.ui.enable_window,
            place_outside_capture=config.ui.place_outside_capture,
        )
    )
    pipeline = MainPipeline(
        screen_capture=screen_capture,
        card_detector=card_detector,
        state_normalizer=state_normalizer,
        policy_forward=rl_forward,
        overlay_renderer=overlay_renderer,
        frame_budget_ms=config.pipeline.frame_budget_ms,
    )
    runtime = AssistantRuntime(
        config=config,
        pipeline=pipeline,
        screen_capture=screen_capture,
        card_detector=card_detector,
        rl_forward=rl_forward,
        overlay_renderer=overlay_renderer,
    )
    return runtime


async def run_once_from_config(config_path: str = "config/default.json") -> PipelineSnapshot:
    config = load_config(config_path)
    runtime = build_runtime(config)
    try:
        snapshot = await runtime.run_once()
    finally:
        runtime.close()
    return snapshot


def format_snapshot_status(snapshot: PipelineSnapshot) -> str:
    frame_size = f"{snapshot.frame.width}x{snapshot.frame.height}"
    rect = snapshot.frame.window_rect
    rect_text = f"({rect.left},{rect.top},{rect.width},{rect.height})"
    self_seen = _format_rank_counts(_count_detection_ranks(snapshot, "self"))
    table_seen = _format_rank_counts(_count_detection_ranks(snapshot, "table"))
    self_hand = _format_rank_counts(snapshot.state_view.self_rank_counts)
    top_action = "none"
    top_reason = "none"
    if len(snapshot.recommendations) > 0:
        top_recommendation = snapshot.recommendations[0]
        top_action = _format_ranks(top_recommendation.action.ranks)
        top_reason = top_recommendation.reason_code
    status = (
        f"frame={snapshot.frame.frame_id} size={frame_size} rect={rect_text} "
        f"detections={len(snapshot.detections)} recommendations={len(snapshot.recommendations)} "
        f"turn={snapshot.state_view.current_turn.value} self_seen={self_seen} table_seen={table_seen} self_hand={self_hand} "
        f"top_action={top_action} reason={top_reason} render={snapshot.render_frame.text_blocks} over_budget={snapshot.over_budget} "
        f"steps={snapshot.elapsed_ms_by_step}"
    )
    return status


def _count_detection_ranks(snapshot: PipelineSnapshot, seat_region: str) -> dict:
    counts = {}
    for detection in snapshot.detections:
        if detection.seat_region.value == seat_region:
            counts[detection.card_rank] = counts.get(detection.card_rank, 0) + 1
    return counts


def _format_rank_counts(rank_counts: dict) -> str:
    pieces: list[str] = []
    for rank, count in sorted(rank_counts.items(), key=lambda item: int(item[0])):
        pieces.append(f"{_rank_symbol(rank)}x{count}")
    text = ",".join(pieces)
    if text == "":
        text = "-"
    return text


def _format_ranks(ranks: tuple) -> str:
    text = ",".join(_rank_symbol(rank) for rank in ranks)
    if text == "":
        text = "过"
    return text


def _rank_symbol(rank: object) -> str:
    symbols = {
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "TEN": "10",
        "JACK": "J",
        "QUEEN": "Q",
        "KING": "K",
        "ACE": "A",
        "TWO": "2",
        "SMALL_JOKER": "小王",
        "BIG_JOKER": "大王",
    }
    name = getattr(rank, "name", str(rank))
    text = symbols.get(name, name)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Doudizhu assistant pipeline.")
    parser.add_argument("--config", default="config/default.json", help="Path to config JSON.")
    parser.add_argument("--once", action="store_true", help="Run one frame and exit.")
    parser.add_argument("--frames", type=int, default=0, help="Run a fixed number of frames; 0 means continuous.")
    parser.add_argument("--status-interval", type=float, default=1.0, help="Seconds between status logs.")
    parser.add_argument("--debug-windows", action="store_true", help="Print visible window titles and exit.")
    parser.add_argument("--health", action="store_true", help="Print runtime health diagnostics and exit.")
    parser.add_argument("--save-frame", type=Path, default=None, help="Save the captured frame as a PNG and exit.")
    parser.add_argument("--save-detections", type=Path, default=None, help="Save captured frame with detection boxes and exit.")
    parser.add_argument("--bring-window-front", action="store_true", help="Bring the configured game window to the foreground before running.")
    args = parser.parse_args()
    return args


async def run_from_args(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    runtime = build_runtime(config)
    try:
        if args.bring_window_front:
            brought_to_front = runtime.screen_capture.bring_window_to_front()
            print(f"bring_window_front={brought_to_front}", flush=True)
        if args.debug_windows:
            titles = runtime.screen_capture.list_visible_window_titles()
            print("Visible window titles:", flush=True)
            for title in titles:
                print(f"- {title}", flush=True)
        elif args.health:
            print_runtime_health(runtime)
        elif args.save_frame is not None:
            snapshot = await runtime.run_once()
            save_frame_png(snapshot, args.save_frame)
            print(format_snapshot_status(snapshot), flush=True)
            print(f"Saved frame: {args.save_frame}", flush=True)
        elif args.save_detections is not None:
            snapshot = await runtime.run_once()
            save_detection_png(snapshot, args.save_detections)
            print(format_snapshot_status(snapshot), flush=True)
            print(f"Saved detections: {args.save_detections}", flush=True)
        elif args.once:
            snapshot = await runtime.run_once()
            print(format_snapshot_status(snapshot), flush=True)
        elif args.frames > 0:
            snapshots = await runtime.run_frames(args.frames)
            for snapshot in snapshots:
                print(format_snapshot_status(snapshot), flush=True)
        else:
            print("Running assistant loop. Press Ctrl+C to stop.", flush=True)
            await runtime.run_loop(args.status_interval)
    finally:
        runtime.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_from_args(args))
    except KeyboardInterrupt:
        print("Assistant stopped.", flush=True)


def print_runtime_health(runtime: AssistantRuntime) -> None:
    print("Runtime health:", flush=True)
    print(f"- window_title: {runtime.config.window_title}", flush=True)
    print(f"- capture_fps: {runtime.config.capture_fps}", flush=True)
    print(f"- card_detector_loaded: {runtime.card_detector.is_loaded()}", flush=True)
    print(f"- card_detector_backend: {runtime.card_detector.backend_name()}", flush=True)
    print(f"- card_detector_model: {runtime.config.models.card_detector_path}", flush=True)
    print(f"- rl_model_loaded: {runtime.rl_forward.is_loaded()}", flush=True)
    print(f"- rl_expected_input_size: {runtime.rl_forward.expected_input_size()}", flush=True)
    print(f"- rl_model: {runtime.config.models.policy_model_path}", flush=True)
    print(f"- overlay_enabled: {runtime.config.ui.enable_window}", flush=True)
    print(f"- overlay_outside_capture: {runtime.config.ui.place_outside_capture}", flush=True)
    print(f"- class_names: {runtime.card_detector.class_names()}", flush=True)


def save_frame_png(snapshot: PipelineSnapshot, output_path: Path) -> None:
    image = build_frame_image(snapshot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_detection_png(snapshot: PipelineSnapshot, output_path: Path) -> None:
    image = build_frame_image(snapshot)
    draw = ImageDraw.Draw(image)
    for detection in snapshot.detections:
        left = detection.bbox.x
        top = detection.bbox.y
        right = detection.bbox.x + detection.bbox.width
        bottom = detection.bbox.y + detection.bbox.height
        label = f"{detection.card_rank.name} {detection.confidence:.2f} {detection.seat_region.value}"
        draw.rectangle((left, top, right, bottom), outline="#00FF66", width=3)
        draw.text((left, max(0, top - 14)), label, fill="#00FF66")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_frame_image(snapshot: PipelineSnapshot) -> Image.Image:
    if snapshot.frame.width <= 0 or snapshot.frame.height <= 0:
        raise RuntimeError("Cannot save frame because no window was captured.")
    expected_length = snapshot.frame.width * snapshot.frame.height * 3
    if len(snapshot.frame.pixels) != expected_length:
        raise RuntimeError("Cannot save frame because pixel data length is invalid.")
    bgr = np.frombuffer(snapshot.frame.pixels, dtype=np.uint8).reshape((snapshot.frame.height, snapshot.frame.width, 3))
    rgb = bgr[:, :, ::-1]
    image = Image.fromarray(rgb)
    return image


if __name__ == "__main__":
    main()
