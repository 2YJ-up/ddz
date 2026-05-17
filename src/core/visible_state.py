from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.core.cards import normalize_cards

PlayerPosition = Literal["landlord", "landlord_down", "landlord_up"]

PLAYER_POSITIONS: tuple[str, ...] = ("landlord", "landlord_down", "landlord_up")
NEXT_POSITION: dict[str, str] = {
    "landlord": "landlord_down",
    "landlord_down": "landlord_up",
    "landlord_up": "landlord",
}


@dataclass(frozen=True, slots=True)
class VisibleAction:
    player: PlayerPosition
    cards: list[str] = field(default_factory=list)
    is_pass: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cards", normalize_cards(self.cards))
        if self.is_pass and len(self.cards) > 0:
            raise ValueError("pass action cannot contain cards")

    def to_dict(self) -> dict[str, object]:
        return {
            "player": self.player,
            "cards": list(self.cards),
            "is_pass": bool(self.is_pass),
        }


@dataclass(frozen=True, slots=True)
class VisibleGameState:
    self_position: PlayerPosition
    acting_player: PlayerPosition
    self_hand: list[str]
    public_action_history: list[VisibleAction]
    num_cards_left: dict[str, int]
    landlord_public_cards: list[str] = field(default_factory=list)
    recognition_confidence: float = 1.0
    last_non_pass_action: VisibleAction | None = None
    frame_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "self_hand", normalize_cards(self.self_hand))
        object.__setattr__(self, "landlord_public_cards", normalize_cards(self.landlord_public_cards))
        object.__setattr__(self, "num_cards_left", {str(key): int(value) for key, value in self.num_cards_left.items()})

        history = [
            action if isinstance(action, VisibleAction) else VisibleAction(**action)  # type: ignore[arg-type]
            for action in self.public_action_history
        ]
        object.__setattr__(self, "public_action_history", history)

        last_non_pass = self.last_non_pass_action
        if last_non_pass is None:
            pass_count = 0
            for action in history:
                if action.is_pass:
                    if last_non_pass is not None:
                        pass_count += 1
                        if pass_count >= 2:
                            last_non_pass = None
                            pass_count = 0
                else:
                    last_non_pass = action
                    pass_count = 0
        elif not isinstance(last_non_pass, VisibleAction):
            last_non_pass = VisibleAction(**last_non_pass)  # type: ignore[arg-type]
        object.__setattr__(self, "last_non_pass_action", last_non_pass)

    def to_dict(self) -> dict[str, object]:
        return {
            "self_position": self.self_position,
            "acting_player": self.acting_player,
            "self_hand": list(self.self_hand),
            "public_action_history": [action.to_dict() for action in self.public_action_history],
            "num_cards_left": dict(self.num_cards_left),
            "landlord_public_cards": list(self.landlord_public_cards),
            "recognition_confidence": float(self.recognition_confidence),
            "last_non_pass_action": self.last_non_pass_action.to_dict() if self.last_non_pass_action else None,
            "frame_id": self.frame_id,
        }


def make_visible_action(raw: object) -> VisibleAction:
    if isinstance(raw, VisibleAction):
        return raw
    if not isinstance(raw, dict):
        raise ValueError("action must be an object")
    return VisibleAction(
        player=raw.get("player"),  # type: ignore[arg-type]
        cards=raw.get("cards") or [],
        is_pass=bool(raw.get("is_pass", False)),
    )
