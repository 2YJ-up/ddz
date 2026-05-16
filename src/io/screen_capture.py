from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mss
import numpy as np
import win32gui

from src.domain.actions import FrameBuffer, WindowRect


@dataclass(frozen=True, slots=True)
class ScreenCaptureConfig:
    window_title: str


class ScreenCapture:
    def __init__(self, config: ScreenCaptureConfig) -> None:
        self.config = config
        self._capture_device = self._create_capture_device()

    def capture(self) -> FrameBuffer:
        hwnd = win32gui.FindWindow(None, self.config.window_title)
        timestamp_ms = self._timestamp_ms()
        window_rect = self._blank_window_rect()
        pixels = b""
        width = 0
        height = 0

        if hwnd != 0:
            window_rect = self._get_client_window_rect(hwnd)
            width = window_rect.width
            height = window_rect.height

            if width > 0 and height > 0:
                timestamp_ms = self._timestamp_ms()
                pixels = self._grab_bgr_pixels(window_rect)
                if len(pixels) != width * height * 3:
                    window_rect = self._blank_window_rect()
                    pixels = b""
                    width = 0
                    height = 0

        frame = self._build_frame(
            pixels=pixels,
            width=width,
            height=height,
            timestamp_ms=timestamp_ms,
            window_rect=window_rect,
        )
        return frame

    def close(self) -> None:
        self._capture_device.close()

    def _get_client_window_rect(self, hwnd: int) -> WindowRect:
        rect = self._blank_window_rect()
        is_visible = win32gui.IsWindowVisible(hwnd) != 0
        is_minimized = win32gui.IsIconic(hwnd) != 0

        if is_visible and not is_minimized:
            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
            top_left = win32gui.ClientToScreen(hwnd, (client_left, client_top))
            bottom_right = win32gui.ClientToScreen(hwnd, (client_right, client_bottom))
            width = max(0, bottom_right[0] - top_left[0])
            height = max(0, bottom_right[1] - top_left[1])
            rect = WindowRect(left=top_left[0], top=top_left[1], width=width, height=height)

        return rect

    def _grab_bgr_pixels(self, window_rect: WindowRect) -> bytes:
        monitor = {
            "left": window_rect.left,
            "top": window_rect.top,
            "width": window_rect.width,
            "height": window_rect.height,
        }
        pixels = b""
        try:
            screenshot = self._capture_device.grab(monitor)
            bgra_array = np.frombuffer(screenshot.bgra, dtype=np.uint8).reshape(
                (window_rect.height, window_rect.width, 4)
            )
            bgr_array = np.ascontiguousarray(bgra_array[:, :, :3])
            pixels = bgr_array.tobytes()
        except (OSError, ValueError, mss.exception.ScreenShotError):
            pixels = b""
        return pixels

    def _build_frame(
        self,
        pixels: bytes,
        width: int,
        height: int,
        timestamp_ms: int,
        window_rect: WindowRect,
    ) -> FrameBuffer:
        frame = FrameBuffer(
            pixels=pixels,
            width=width,
            height=height,
            timestamp_ms=timestamp_ms,
            window_rect=window_rect,
            frame_id=f"{self.config.window_title}:{timestamp_ms}",
        )
        return frame

    def _timestamp_ms(self) -> int:
        timestamp_ms = time.time_ns() // 1_000_000
        return timestamp_ms

    def _blank_window_rect(self) -> WindowRect:
        rect = WindowRect(left=0, top=0, width=0, height=0)
        return rect

    def _create_capture_device(self) -> Any:
        factory = getattr(mss, "MSS", mss.mss)
        capture_device = factory()
        return capture_device
