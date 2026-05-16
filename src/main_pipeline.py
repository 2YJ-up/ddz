from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Protocol

from src.domain.actions import ActionRecommendation, CardDetection, FrameBuffer, GameStateView, RenderFrame, WindowRect


class ScreenCaptureProtocol(Protocol):
    def capture(self) -> FrameBuffer:
        ...


class CardDetectorProtocol(Protocol):
    def detect(self, frame: FrameBuffer) -> tuple[CardDetection, ...]:
        ...


class StateNormalizerProtocol(Protocol):
    def normalize(self, detections: tuple[CardDetection, ...]) -> GameStateView:
        ...


class PolicyForwardProtocol(Protocol):
    def recommend(self, state_view: GameStateView) -> tuple[ActionRecommendation, ...]:
        ...


class OverlayRendererProtocol(Protocol):
    def render(
        self,
        recommendations: tuple[ActionRecommendation, ...],
        game_window_rect: WindowRect,
    ) -> RenderFrame:
        ...


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    frame: FrameBuffer
    detections: tuple[CardDetection, ...]
    state_view: GameStateView
    recommendations: tuple[ActionRecommendation, ...]
    render_frame: RenderFrame
    elapsed_ms_by_step: dict[str, float]
    over_budget: bool


class MainPipeline:
    def __init__(
        self,
        screen_capture: ScreenCaptureProtocol,
        card_detector: CardDetectorProtocol,
        state_normalizer: StateNormalizerProtocol,
        policy_forward: PolicyForwardProtocol,
        overlay_renderer: OverlayRendererProtocol,
        frame_budget_ms: int,
    ) -> None:
        self.screen_capture = screen_capture
        self.card_detector = card_detector
        self.state_normalizer = state_normalizer
        self.policy_forward = policy_forward
        self.overlay_renderer = overlay_renderer
        self.frame_budget_ms = frame_budget_ms

    async def run_once(self) -> PipelineSnapshot:
        elapsed_ms_by_step: dict[str, float] = {}
        total_start = time.perf_counter()

        step_start = time.perf_counter()
        frame = await self._maybe_await(self.screen_capture.capture())
        elapsed_ms_by_step["IO_READ"] = self._elapsed_ms(step_start)

        step_start = time.perf_counter()
        detections = await self._maybe_await(self.card_detector.detect(frame))
        elapsed_ms_by_step["CV_INFERENCE"] = self._elapsed_ms(step_start)

        step_start = time.perf_counter()
        state_view = await self._maybe_await(self.state_normalizer.normalize(detections))
        elapsed_ms_by_step["STATE_NORMALIZATION"] = self._elapsed_ms(step_start)

        step_start = time.perf_counter()
        recommendations = await self._maybe_await(self.policy_forward.recommend(state_view))
        elapsed_ms_by_step["RL_FORWARD"] = self._elapsed_ms(step_start)

        step_start = time.perf_counter()
        render_frame = await self._maybe_await(self.overlay_renderer.render(recommendations, frame.window_rect))
        elapsed_ms_by_step["UI_RENDER"] = self._elapsed_ms(step_start)

        total_elapsed = self._elapsed_ms(total_start)
        snapshot = PipelineSnapshot(
            frame=frame,
            detections=detections,
            state_view=state_view,
            recommendations=recommendations,
            render_frame=render_frame,
            elapsed_ms_by_step=elapsed_ms_by_step,
            over_budget=total_elapsed > float(self.frame_budget_ms),
        )
        return snapshot

    async def _maybe_await(self, value: Any) -> Any:
        result = value
        if inspect.isawaitable(value):
            result = await value
        return result

    def _elapsed_ms(self, start: float) -> float:
        elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed
