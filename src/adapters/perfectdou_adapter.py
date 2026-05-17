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
    last_two_moves: list[list[int]]
    last_move_dict: dict[str, list[int]]
    played_cards: dict[str, list[int]]
    num_cards_left: dict[str, int]
    num_cards_left_dict: dict[str, int]
    landlord_public_cards: list[int]
    three_landlord_cards: list[int]
    card_play_action_seq: list[list[int]]
    legal_actions: list[list[int]]
    bomb_num: int = 0
    last_pid: str = "landlord"
    all_handcards: dict[str, list[int]] = field(default_factory=dict)
    sample_hands: dict[str, list[int]] = field(default_factory=dict)


class PerfectDouAdapter:
    def __init__(self, repo_path: str = "external/PerfectDou", model_name: str = "perfectdou") -> None:
        self.repo_path = Path(repo_path)
        self.model_name = model_name
        self._import_error: str = ""
        self._rules_error: str = ""
        self._policy_error: str = ""
        self._module: Any = None
        self._moves_gener_cls: Any = None
        self._move_detector: Any = None
        self._move_selector: Any = None
        self._agent_cls: Any = None
        self._agents: dict[str, Any] = {}
        self._load_repo_if_available()

    @property
    def available(self) -> bool:
        return self._module is not None

    @property
    def rules_available(self) -> bool:
        return self._moves_gener_cls is not None and self._move_detector is not None and self._move_selector is not None

    @property
    def policy_available(self) -> bool:
        return self._agent_cls is not None

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        if not self.repo_path.exists():
            return f"PerfectDou repo not found at {self.repo_path}"
        return self._import_error or "PerfectDou could not be imported"

    @property
    def rules_unavailable_reason(self) -> str:
        if self.rules_available:
            return ""
        return self._rules_error or self.unavailable_reason or "PerfectDou rules could not be imported"

    @property
    def policy_unavailable_reason(self) -> str:
        if self.policy_available:
            return ""
        return self._policy_error or self.unavailable_reason or "PerfectDou policy could not be imported"

    def to_env_cards(self, cards: list[str]) -> list[int]:
        return [int(PERFECTDOU_CARD_MAP[card]) for card in normalize_cards(cards)]

    def from_env_cards(self, cards: list[int]) -> list[str]:
        env_to_card = {int(value): key for key, value in PERFECTDOU_CARD_MAP.items()}
        converted = [env_to_card[int(card)] for card in cards]
        return normalize_cards(converted)

    def build_infoset(self, state: VisibleGameState, sample: object | None = None) -> PerfectDouInfoSet:
        from src.core.belief_sampler import build_unknown_pool

        sample_hands: dict[str, list[int]] = {}
        raw_hands: dict[str, list[str]] = {}
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
        last_move_dict = {player: [] for player in PLAYER_POSITIONS}
        card_play_action_seq: list[list[int]] = []
        last_pid = "landlord"
        for action in state.public_action_history:
            env_action = self.to_env_cards(action.cards)
            played_cards[action.player].extend(env_action)
            last_move_dict[action.player] = list(env_action)
            card_play_action_seq.append(list(env_action))
            if len(env_action) > 0:
                last_pid = action.player

        last_move = state.last_non_pass_action.cards if state.last_non_pass_action is not None else []
        all_handcards: dict[str, list[int]] = {state.self_position: self.to_env_cards(state.self_hand)}
        for player, cards in raw_hands.items():
            all_handcards[player] = self.to_env_cards(cards)
        legal_actions = self.generate_legal_actions_env(state)
        return PerfectDouInfoSet(
            player_position=state.self_position,
            player_hand_cards=self.to_env_cards(state.self_hand),
            other_hand_cards=self.to_env_cards(outside_cards),
            last_move=self.to_env_cards(last_move),
            last_two_moves=_last_two_moves(card_play_action_seq),
            last_move_dict=last_move_dict,
            played_cards=played_cards,
            num_cards_left=dict(state.num_cards_left),
            num_cards_left_dict=dict(state.num_cards_left),
            landlord_public_cards=self.to_env_cards(state.landlord_public_cards),
            three_landlord_cards=self.to_env_cards(state.landlord_public_cards),
            card_play_action_seq=card_play_action_seq,
            legal_actions=legal_actions,
            bomb_num=_count_bombs(card_play_action_seq),
            last_pid=last_pid,
            all_handcards=all_handcards,
            sample_hands=sample_hands,
        )

    def generate_legal_actions(self, state: VisibleGameState) -> list[list[str]]:
        if self.rules_available:
            try:
                actions = [self.from_env_cards(action) for action in self.generate_legal_actions_env(state)]
            except Exception as exc:
                self._rules_error = str(exc)
                actions = generate_legal_actions(state)
        else:
            actions = generate_legal_actions(state)
        return _dedupe_card_actions(actions)

    def generate_legal_actions_env(self, state: VisibleGameState) -> list[list[int]]:
        if not self.rules_available:
            return [self.to_env_cards(action) for action in generate_legal_actions(state)]
        hand = self.to_env_cards(state.self_hand)
        rival_move = []
        if state.last_non_pass_action is not None:
            rival_move = self.to_env_cards(state.last_non_pass_action.cards)
        moves = self._official_legal_moves(hand, rival_move)
        return _dedupe_env_actions(moves)

    def recommend_action(self, state: VisibleGameState, sample: object | None = None) -> list[str]:
        if not self.policy_available:
            return []
        infoset = self.build_infoset(state, sample)
        try:
            agent = self._agent_for_position(state.self_position)
            action = agent.act(infoset)
            result = self.from_env_cards(action)
        except Exception as exc:
            self._policy_error = str(exc)
            result = []
        return result

    def _load_repo_if_available(self) -> None:
        if not self.repo_path.exists():
            return
        repo_str = str(self.repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            import perfectdou  # type: ignore[import-not-found]

            self._module = perfectdou
        except Exception as exc:  # pragma: no cover - depends on local checkout
            self._import_error = str(exc)
            self._module = None
        self._load_rule_modules()
        self._load_policy_agent()

    def _load_rule_modules(self) -> None:
        if self._module is None:
            return
        try:
            from perfectdou.env import move_detector, move_selector
            from perfectdou.env.move_generator import MovesGener

            self._moves_gener_cls = MovesGener
            self._move_detector = move_detector
            self._move_selector = move_selector
        except Exception as exc:  # pragma: no cover - depends on external repo
            self._rules_error = str(exc)
            self._moves_gener_cls = None
            self._move_detector = None
            self._move_selector = None

    def _load_policy_agent(self) -> None:
        if self._module is None:
            return
        try:
            from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent

            self._agent_cls = PerfectDouAgent
        except Exception as exc:  # pragma: no cover - platform-specific encode extension
            self._policy_error = str(exc)
            self._agent_cls = None

    def _agent_for_position(self, position: str) -> Any:
        agent = self._agents.get(position)
        if agent is None and self._agent_cls is not None:
            agent = self._agent_cls(position)
            self._agents[position] = agent
        return agent

    def _official_legal_moves(self, hand: list[int], rival_move: list[int]) -> list[list[int]]:
        md = self._move_detector
        ms = self._move_selector
        mg = self._moves_gener_cls(hand)
        rival_type = md.get_move_type(list(rival_move))
        rival_move_type = rival_type["type"]
        rival_move_len = rival_type.get("len", 1)

        if rival_move_type == md.TYPE_0_PASS:
            moves = mg.gen_moves()
        elif rival_move_type == md.TYPE_1_SINGLE:
            moves = ms.filter_type_1_single(mg.gen_type_1_single(), list(rival_move))
        elif rival_move_type == md.TYPE_2_PAIR:
            moves = ms.filter_type_2_pair(mg.gen_type_2_pair(), list(rival_move))
        elif rival_move_type == md.TYPE_3_TRIPLE:
            moves = ms.filter_type_3_triple(mg.gen_type_3_triple(), list(rival_move))
        elif rival_move_type == md.TYPE_4_BOMB:
            moves = ms.filter_type_4_bomb(mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), list(rival_move))
        elif rival_move_type == md.TYPE_5_KING_BOMB:
            moves = []
        elif rival_move_type == md.TYPE_6_3_1:
            moves = ms.filter_type_6_3_1(mg.gen_type_6_3_1(), list(rival_move))
        elif rival_move_type == md.TYPE_7_3_2:
            moves = ms.filter_type_7_3_2(mg.gen_type_7_3_2(), list(rival_move))
        elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
            moves = ms.filter_type_8_serial_single(mg.gen_type_8_serial_single(repeat_num=rival_move_len), list(rival_move))
        elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
            moves = ms.filter_type_9_serial_pair(mg.gen_type_9_serial_pair(repeat_num=rival_move_len), list(rival_move))
        elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
            moves = ms.filter_type_10_serial_triple(mg.gen_type_10_serial_triple(repeat_num=rival_move_len), list(rival_move))
        elif rival_move_type == md.TYPE_11_SERIAL_3_1:
            moves = ms.filter_type_11_serial_3_1(mg.gen_type_11_serial_3_1(repeat_num=rival_move_len), list(rival_move))
        elif rival_move_type == md.TYPE_12_SERIAL_3_2:
            moves = ms.filter_type_12_serial_3_2(mg.gen_type_12_serial_3_2(repeat_num=rival_move_len), list(rival_move))
        elif rival_move_type == md.TYPE_13_4_2:
            moves = ms.filter_type_13_4_2(mg.gen_type_13_4_2(), list(rival_move))
        elif rival_move_type == md.TYPE_14_4_22:
            moves = ms.filter_type_14_4_22(mg.gen_type_14_4_22(), list(rival_move))
        else:
            moves = []

        should_add_bombs = rival_move_type not in {md.TYPE_0_PASS, md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB}
        if should_add_bombs:
            moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
        if len(rival_move) != 0:
            moves = moves + [[]]
        return [sorted(move) for move in moves]


def _dedupe_env_actions(actions: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    deduped: list[list[int]] = []
    for action in actions:
        key = tuple(action)
        is_new = key not in seen
        if is_new:
            seen.add(key)
            deduped.append(list(action))
    return deduped


def _dedupe_card_actions(actions: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    for action in actions:
        key = tuple(action)
        is_new = key not in seen
        if is_new:
            seen.add(key)
            deduped.append(list(action))
    return deduped


def _last_two_moves(action_seq: list[list[int]]) -> list[list[int]]:
    recent = list(reversed(action_seq[-2:]))
    while len(recent) < 2:
        recent.append([])
    return recent[:2]


def _count_bombs(action_seq: list[list[int]]) -> int:
    bomb_count = 0
    for action in action_seq:
        counts = {card: action.count(card) for card in set(action)}
        is_rank_bomb = len(action) == 4 and len(counts) == 1
        is_rocket = action == [20, 30]
        if is_rank_bomb or is_rocket:
            bomb_count += 1
    return bomb_count
