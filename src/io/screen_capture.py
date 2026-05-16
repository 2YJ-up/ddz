from __future__ import annotations

import time
from dataclasses import dataclass

from src.domain.actions import FrameBuffer, WindowRect


@dataclass(frozen=True, slots=True)
class ScreenCaptureConfig:
    window_title: str


class ScreenCapture:
    def __init__(self, config: ScreenCaptureConfig) -> None:
        self.config = config

    def capture(self) -> FrameBuffer:
        timestamp_ms = int(time.time() * 1000)
        frame = FrameBuffer(
            pixels=b"",
            width=0,
            height=0,
            timestamp_ms=timestamp_ms,
            window_rect=WindowRect(left=0, top=0, width=0, height=0),
            frame_id=f"{self.config.window_title}:{timestamp_ms}",
        )
        return frame
