from __future__ import annotations

import importlib
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
        index = 0
        while index < len(recommendations):
            recommendation = recommendations[index]
            card_text = ",".join(rank.name for rank in recommendation.action.ranks)
            text_blocks.append(
                f"{index + 1}. {card_text} p={recommendation.probability:.2f} win={recommendation.expected_win_rate:.2f}"
            )
            if index == 0:
                highlight_ranks = recommendation.action.ranks
            index += 1

        anchor_rect = WindowRect(
            left=game_window_rect.left + self.config.offset_x,
            top=game_window_rect.top + self.config.offset_y,
            width=self.config.width,
            height=self.config.height,
        )
        frame = RenderFrame(
            text_blocks=tuple(text_blocks),
            highlight_ranks=highlight_ranks,
            anchor_rect=anchor_rect,
        )
        if self.config.enable_window:
            self._render_window(frame)
        return frame

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
