from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.core.cards import normalize_cards
from src.core.visible_state import PLAYER_POSITIONS, VisibleAction, VisibleGameState


@dataclass(frozen=True, slots=True)
class RecognizedFrameState:
    self_hand: list[str]
    last_table_action: VisibleAction | None
    public_action_history: list[VisibleAction]
    player_position: str
    acting_player: str
    num_cards_left: dict[str, int]
    landlord_public_cards: list[str]
    recognition_confidence: float
    frame_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_visible_state(self) -> VisibleGameState:
        history = list(self.public_action_history)
        if not history and self.last_table_action is not None:
            history = [self.last_table_action]
        return VisibleGameState(
            self_position=self.player_position,  # type: ignore[arg-type]
            acting_player=self.acting_player,  # type: ignore[arg-type]
            self_hand=list(self.self_hand),
            public_action_history=history,
            num_cards_left=dict(self.num_cards_left),
            landlord_public_cards=list(self.landlord_public_cards),
            recognition_confidence=float(self.recognition_confidence),
            frame_id=self.frame_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "self_hand": list(self.self_hand),
            "last_table_action": self.last_table_action.to_dict() if self.last_table_action else None,
            "public_action_history": [action.to_dict() for action in self.public_action_history],
            "player_position": self.player_position,
            "acting_player": self.acting_player,
            "num_cards_left": dict(self.num_cards_left),
            "landlord_public_cards": list(self.landlord_public_cards),
            "recognition_confidence": self.recognition_confidence,
            "frame_id": self.frame_id,
            "warnings": list(self.warnings),
        }


def load_recognized_frame_json(path: str | Path) -> RecognizedFrameState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_recognized_frame_state(raw)


def parse_recognized_frame_state(raw: Mapping[str, Any]) -> RecognizedFrameState:
    warnings = [str(item) for item in raw.get("warnings", [])]

    self_hand = _cards_or_empty(raw, "self_hand", warnings)
    last_table_action = _optional_action(raw.get("last_table_action"), "last_table_action", warnings)
    public_action_history = _action_history(raw.get("public_action_history"), warnings)
    player_position = _position_or_default(raw.get("player_position"), "player_position", warnings)
    acting_player = _position_or_default(raw.get("acting_player"), "acting_player", warnings, default=player_position)
    num_cards_left = _num_cards_left(raw.get("num_cards_left"), warnings)
    landlord_public_cards = _cards_or_empty(raw, "landlord_public_cards", warnings, missing_ok=True)
    recognition_confidence = _confidence(raw.get("recognition_confidence"), warnings)
    frame_id = "" if raw.get("frame_id") is None else str(raw.get("frame_id", ""))

    if raw.get("self_hand") is None:
        warnings.append("self_hand was null or missing; recommendation will be low confidence")
    if raw.get("public_action_history") is None:
        warnings.append("public_action_history was null or missing; last_table_action will be used if available")
    if raw.get("num_cards_left") is None:
        warnings.append("num_cards_left was null or missing")

    return RecognizedFrameState(
        self_hand=self_hand,
        last_table_action=last_table_action,
        public_action_history=public_action_history,
        player_position=player_position,
        acting_player=acting_player,
        num_cards_left=num_cards_left,
        landlord_public_cards=landlord_public_cards,
        recognition_confidence=recognition_confidence,
        frame_id=frame_id,
        warnings=warnings,
    )


def _cards_or_empty(raw: Mapping[str, Any], key: str, warnings: list[str], *, missing_ok: bool = False) -> list[str]:
    value = raw.get(key)
    if value is None:
        if not missing_ok:
            warnings.append(f"{key} is missing")
        return []
    try:
        return normalize_cards(value)
    except (TypeError, ValueError) as exc:
        warnings.append(f"{key} could not be normalized: {exc}")
        return []


def _optional_action(value: Any, field_name: str, warnings: list[str]) -> VisibleAction | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        warnings.append(f"{field_name} is not an object")
        return None
    try:
        return VisibleAction(
            player=_position_or_default(value.get("player"), f"{field_name}.player", warnings),
            cards=normalize_cards(value.get("cards") or []),
            is_pass=bool(value.get("is_pass", False)),
        )
    except ValueError as exc:
        warnings.append(f"{field_name} is invalid: {exc}")
        return None


def _action_history(value: Any, warnings: list[str]) -> list[VisibleAction]:
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append("public_action_history is not a list")
        return []
    actions: list[VisibleAction] = []
    for index, item in enumerate(value):
        action = _optional_action(item, f"public_action_history[{index}]", warnings)
        if action is not None:
            actions.append(action)
    return actions


def _position_or_default(
    value: Any,
    field_name: str,
    warnings: list[str],
    *,
    default: str = "landlord",
) -> str:
    text = default if value is None else str(value)
    if text not in PLAYER_POSITIONS:
        warnings.append(f"{field_name}={text!r} is invalid; defaulting to {default!r}")
        text = default
    return text


def _num_cards_left(value: Any, warnings: list[str]) -> dict[str, int]:
    result = {player: 0 for player in PLAYER_POSITIONS}
    if not isinstance(value, Mapping):
        warnings.append("num_cards_left is not an object")
        return result
    for player in PLAYER_POSITIONS:
        try:
            result[player] = int(value.get(player, 0))
        except (TypeError, ValueError):
            warnings.append(f"num_cards_left[{player}] is invalid; defaulting to 0")
            result[player] = 0
    return result


def _confidence(value: Any, warnings: list[str]) -> float:
    if value is None:
        warnings.append("recognition_confidence is missing; defaulting to 0")
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        warnings.append("recognition_confidence is invalid; defaulting to 0")
        confidence = 0.0
    return max(0.0, min(1.0, confidence))
