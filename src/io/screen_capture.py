from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mss
import numpy as np
import win32con
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
        hwnd = self._find_window_handle()
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

    def list_visible_window_titles(self) -> tuple[str, ...]:
        titles: list[str] = []

        def collect_title(hwnd: int, _: object) -> bool:
            if win32gui.IsWindowVisible(hwnd) != 0:
                title = win32gui.GetWindowText(hwnd).strip()
                if title != "":
                    titles.append(title)
            return True

        win32gui.EnumWindows(collect_title, None)
        result = tuple(titles)
        return result

    def close(self) -> None:
        self._capture_device.close()

    def bring_window_to_front(self) -> bool:
        hwnd = self._find_window_handle()
        brought_to_front = False
        if hwnd != 0:
            try:
                flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    self._ignore_foreground_denial()
                brought_to_front = True
            except Exception:
                brought_to_front = False
        return brought_to_front

    def _ignore_foreground_denial(self) -> None:
        return None

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

    def _find_window_handle(self) -> int:
        exact_hwnd = win32gui.FindWindow(None, self.config.window_title)
        selected_hwnd = exact_hwnd
        if selected_hwnd == 0:
            selected_hwnd = self._find_window_handle_by_title_fragment()
        return selected_hwnd

    def _find_window_handle_by_title_fragment(self) -> int:
        selected_hwnd = 0
        target = self.config.window_title.casefold()

        def inspect_window(hwnd: int, _: object) -> bool:
            nonlocal selected_hwnd
            title = win32gui.GetWindowText(hwnd).strip()
            is_candidate = (
                selected_hwnd == 0
                and target != ""
                and win32gui.IsWindowVisible(hwnd) != 0
                and target in title.casefold()
            )
            if is_candidate:
                selected_hwnd = hwnd
            return True

        win32gui.EnumWindows(inspect_window, None)
        return selected_hwnd
