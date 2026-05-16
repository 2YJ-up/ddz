from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

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


def main() -> None:
    snapshot = asyncio.run(run_once_from_config())
    print(
        {
            "frame": snapshot.frame.frame_id,
            "detections": len(snapshot.detections),
            "recommendations": len(snapshot.recommendations),
            "render_blocks": snapshot.render_frame.text_blocks,
            "over_budget": snapshot.over_budget,
            "elapsed_ms_by_step": snapshot.elapsed_ms_by_step,
        }
    )


if __name__ == "__main__":
    main()
