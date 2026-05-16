from __future__ import annotations

import importlib
import ctypes
from dataclasses import dataclass
from typing import Any

from src.domain.actions import ActionRecommendation, RenderFrame, WindowRect
from src.domain.cards import CardRank


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    offset_x: int
    offset_y: int
    width: int
    height: int
    enable_window: bool = False
    place_outside_capture: bool = True


class OverlayRenderer:
    def __init__(self, config: OverlayConfig) -> None:
        self.config = config
        self._tk: Any = None
        self._root: Any = None
        self._labels: list[Any] = []
        self._window_available = False

    def render(
        self,
        recommendations: tuple[ActionRecommendation, ...],
        game_window_rect: WindowRect,
    ) -> RenderFrame:
        text_blocks: list[str] = []
        highlight_ranks: tuple[CardRank, ...] = ()
        if len(recommendations) > 0 and recommendations[0].risk_text != "":
            text_blocks.append(recommendations[0].risk_text)
        index = 0
        while index < len(recommendations):
            recommendation = recommendations[index]
            card_text = recommendation.action_label
            if card_text == "":
                card_text = self._format_ranks(recommendation.action.ranks)
            reason_text = recommendation.reason_text
            if reason_text != "":
                reason_text = f" {reason_text}"
            text_blocks.append(f"{index + 1}. {card_text} win={recommendation.expected_win_rate:.2f}{reason_text}")
            if index == 0:
                highlight_ranks = recommendation.action.ranks
            index += 1

        if len(text_blocks) == 0:
            text_blocks.append("Waiting for cards...")

        anchor_rect = self._build_anchor_rect(game_window_rect)
        frame = RenderFrame(
            text_blocks=tuple(text_blocks),
            highlight_ranks=highlight_ranks,
            anchor_rect=anchor_rect,
        )
        if self.config.enable_window:
            self._render_window(frame)
        return frame

    def _build_anchor_rect(self, game_window_rect: WindowRect) -> WindowRect:
        screen_width, screen_height = self._screen_size()
        left = game_window_rect.left + self.config.offset_x
        if self.config.place_outside_capture and game_window_rect.width > 0:
            outside_right_left = game_window_rect.left + game_window_rect.width + self.config.offset_x
            outside_left = game_window_rect.left - self.config.width - self.config.offset_x
            inside_left = game_window_rect.left + game_window_rect.width - self.config.width - self.config.offset_x
            if outside_right_left + self.config.width <= screen_width:
                left = outside_right_left
            elif outside_left >= 0:
                left = outside_left
            elif inside_left >= game_window_rect.left:
                left = inside_left
        rect = WindowRect(
            left=self._clamp(left, 0, max(0, screen_width - self.config.width)),
            top=self._clamp(game_window_rect.top + self.config.offset_y, 0, max(0, screen_height - self.config.height)),
            width=self.config.width,
            height=self.config.height,
        )
        return rect

    def _screen_size(self) -> tuple[int, int]:
        width = 1920
        height = 1080
        try:
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
        except (AttributeError, ValueError, OSError):
            width = 1920
            height = 1080
        return width, height

    def _clamp(self, value: int, minimum: int, maximum: int) -> int:
        clamped = value
        if clamped < minimum:
            clamped = minimum
        if clamped > maximum:
            clamped = maximum
        return clamped

    def close(self) -> None:
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                self._root = None
            self._root = None
            self._window_available = False

    def _render_window(self, frame: RenderFrame) -> None:
        self._ensure_window()
        if self._window_available and self._root is not None:
            geometry = f"{frame.anchor_rect.width}x{frame.anchor_rect.height}+{frame.anchor_rect.left}+{frame.anchor_rect.top}"
            self._root.geometry(geometry)
            self._sync_labels(frame.text_blocks)
            self._root.update_idletasks()
            self._root.update()

    def _ensure_window(self) -> None:
        if self._root is None:
            try:
                self._tk = importlib.import_module("tkinter")
                self._root = self._tk.Tk()
                self._root.overrideredirect(True)
                self._root.attributes("-topmost", True)
                self._root.attributes("-alpha", 0.88)
                self._root.configure(bg="#101820")
                self._window_available = True
            except Exception:
                self._root = None
                self._window_available = False

    def _sync_labels(self, text_blocks: tuple[str, ...]) -> None:
        while len(self._labels) < len(text_blocks):
            label = self._tk.Label(
                self._root,
                text="",
                anchor="w",
                justify="left",
                font=("Consolas", 11),
                fg="#E8F5E9",
                bg="#101820",
                padx=10,
                pady=4,
            )
            label.pack(fill="x")
            self._labels.append(label)

        index = 0
        while index < len(self._labels):
            label = self._labels[index]
            if index < len(text_blocks):
                label.configure(text=text_blocks[index])
                label.pack(fill="x")
            else:
                label.pack_forget()
            index += 1

    def _format_ranks(self, ranks: tuple[CardRank, ...]) -> str:
        text = ",".join(self._rank_symbol(rank) for rank in ranks)
        if text == "":
            text = "过"
        return text

    def _rank_symbol(self, rank: CardRank) -> str:
        symbols = {
            CardRank.THREE: "3",
            CardRank.FOUR: "4",
            CardRank.FIVE: "5",
            CardRank.SIX: "6",
            CardRank.SEVEN: "7",
            CardRank.EIGHT: "8",
            CardRank.NINE: "9",
            CardRank.TEN: "10",
            CardRank.JACK: "J",
            CardRank.QUEEN: "Q",
            CardRank.KING: "K",
            CardRank.ACE: "A",
            CardRank.TWO: "2",
            CardRank.SMALL_JOKER: "小王",
            CardRank.BIG_JOKER: "大王",
        }
        text = symbols[rank]
        return text
