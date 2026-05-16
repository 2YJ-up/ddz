from __future__ import annotations

from dataclasses import dataclass

from src.domain.actions import ActionRecommendation, RenderFrame, WindowRect
from src.domain.cards import CardRank


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    offset_x: int
    offset_y: int
    width: int
    height: int


class OverlayRenderer:
    def __init__(self, config: OverlayConfig) -> None:
        self.config = config

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
        return frame
