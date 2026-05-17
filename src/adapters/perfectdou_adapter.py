from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.cards import PERFECTDOU_CARD_MAP, normalize_cards
from src.core.rule_engine import generate_legal_actions
from src.core.visible_state import PLAYER_POSITIONS, VisibleGameState


@dataclass(frozen=True, slots=True)
class PerfectDouInfoSet:
    player_position: str
    player_hand_cards: list[int]
    other_hand_cards: list[int]
    last_move: list[int]
    played_cards: dict[str, list[int]]
    num_cards_left: dict[str, int]
    landlord_public_cards: list[int]
    sample_hands: dict[str, list[int]] = field(default_factory=dict)


class PerfectDouAdapter:
    def __init__(self, repo_path: str = "external/PerfectDou", model_name: str = "perfectdou") -> None:
        self.repo_path = Path(repo_path)
        self.model_name = model_name
        self._import_error: str = ""
        self._module: Any = None
        self._load_repo_if_available()

    @property
    def available(self) -> bool:
        return self._module is not None

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        if not self.repo_path.exists():
            return f"PerfectDou repo not found at {self.repo_path}"
        return self._import_error or "PerfectDou could not be imported"

    def to_env_cards(self, cards: list[str]) -> list[int]:
        return [int(PERFECTDOU_CARD_MAP[card]) for card in normalize_cards(cards)]

    def build_infoset(self, state: VisibleGameState, sample: object | None = None) -> PerfectDouInfoSet:
        from src.core.belief_sampler import build_unknown_pool

        sample_hands: dict[str, list[int]] = {}
        if sample is not None:
            raw_hands = getattr(sample, "hands", {})
            sample_hands = {player: self.to_env_cards(cards) for player, cards in raw_hands.items()}
            outside_cards = []
            for player in PLAYER_POSITIONS:
                if player != state.self_position:
                    outside_cards.extend(raw_hands.get(player, []))
        else:
            outside_cards = build_unknown_pool(state)

        played_cards = {player: [] for player in PLAYER_POSITIONS}
        for action in state.public_action_history:
            played_cards[action.player].extend(self.to_env_cards(action.cards))

        last_move = state.last_non_pass_action.cards if state.last_non_pass_action is not None else []
        return PerfectDouInfoSet(
            player_position=state.self_position,
            player_hand_cards=self.to_env_cards(state.self_hand),
            other_hand_cards=self.to_env_cards(outside_cards),
            last_move=self.to_env_cards(last_move),
            played_cards=played_cards,
            num_cards_left=dict(state.num_cards_left),
            landlord_public_cards=self.to_env_cards(state.landlord_public_cards),
            sample_hands=sample_hands,
        )

    def recommend_action(self, state: VisibleGameState, sample: object | None = None) -> list[str]:
        if not self.available:
            return []
        # The upstream PerfectDou repository has multiple historical entry
        # points. Keep this adapter conservative: construct the InfoSet-like
        # object here and let a future repo-specific bridge consume it without
        # changing the product-facing contract.
        _ = self.build_infoset(state, sample)
        legal_actions = generate_legal_actions(state)
        return legal_actions[0] if legal_actions else []

    def _load_repo_if_available(self) -> None:
        if not self.repo_path.exists():
            return
        repo_str = str(self.repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            # The official package layout is not guaranteed across releases;
            # this import only establishes availability and avoids mutating the
            # third-party repository.
            import perfectdou  # type: ignore[import-not-found]

            self._module = perfectdou
        except Exception as exc:  # pragma: no cover - depends on local checkout
            self._import_error = str(exc)
            self._module = None
