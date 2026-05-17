from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.perfectdou_adapter import PerfectDouAdapter
from src.core.visible_state import VisibleGameState


class DouZeroAdapter:
    def __init__(
        self,
        repo_path: str = "external/PerfectDou",
        model_dir: str = "perfectdou/model/douzero/douzero_ADP",
        rules_adapter: PerfectDouAdapter | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.model_dir = self.repo_path / model_dir
        self.rules_adapter = rules_adapter or PerfectDouAdapter(repo_path=repo_path)
        self._import_error = ""
        self._agent_cls: Any = None
        self._get_obs: Any = None
        self._torch: Any = None
        self._agents: dict[str, Any] = {}
        self._load_modules()

    @property
    def available(self) -> bool:
        models_exist = all((self.model_dir / f"{position}.ckpt").exists() for position in _POSITIONS)
        return self._agent_cls is not None and self._get_obs is not None and self._torch is not None and models_exist

    @property
    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        if not self.repo_path.exists():
            return f"DouZero source not found via {self.repo_path}"
        if not self.model_dir.exists():
            return f"DouZero ADP model directory not found at {self.model_dir}"
        return self._import_error or "DouZero could not be imported"

    def recommend_action(self, state: VisibleGameState, sample: object | None = None) -> list[str]:
        scores = self.score_actions(state, sample)
        action: list[str] = []
        if scores:
            best_key = max(scores, key=lambda key: scores[key])
            action = list(best_key)
        return action

    def score_actions(self, state: VisibleGameState, sample: object | None = None) -> dict[tuple[str, ...], float]:
        if not self.available:
            return {}
        scores: dict[tuple[str, ...], float] = {}
        try:
            infoset = self.rules_adapter.build_infoset(state, sample)
            if len(infoset.legal_actions) == 1:
                only_action = tuple(self.rules_adapter.from_env_cards(infoset.legal_actions[0]))
                scores[only_action] = 1.0
            else:
                agent = self._agent_for_position(state.self_position)
                obs = self._get_obs(infoset)
                z_batch = self._torch.from_numpy(obs["z_batch"]).float()
                x_batch = self._torch.from_numpy(obs["x_batch"]).float()
                if self._torch.cuda.is_available():
                    z_batch = z_batch.cuda()
                    x_batch = x_batch.cuda()
                y_pred = agent.model.forward(z_batch, x_batch, return_value=True)["values"]
                values = y_pred.detach().cpu().numpy().reshape(-1)
                index = 0
                while index < len(obs["legal_actions"]) and index < values.size:
                    action = tuple(self.rules_adapter.from_env_cards(obs["legal_actions"][index]))
                    value = float(values[index])
                    existing = scores.get(action)
                    if np.isfinite(value) and (existing is None or value > existing):
                        scores[action] = value
                    index += 1
        except Exception as exc:
            self._import_error = str(exc)
            scores = {}
        return scores

    def _load_modules(self) -> None:
        if not self.repo_path.exists():
            return
        repo_str = str(self.repo_path.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            import torch
            from perfectdou.env.env import get_obs
            from perfectdou.evaluation.deep_agent import DeepAgent

            self._torch = torch
            self._get_obs = get_obs
            self._agent_cls = DeepAgent
        except Exception as exc:  # pragma: no cover - depends on local ML runtime
            self._import_error = str(exc)
            self._torch = None
            self._get_obs = None
            self._agent_cls = None

    def _agent_for_position(self, position: str) -> Any:
        agent = self._agents.get(position)
        if agent is None and self._agent_cls is not None:
            model_path = self.model_dir / f"{position}.ckpt"
            agent = self._agent_cls(position, str(model_path))
            self._agents[position] = agent
        return agent


_POSITIONS = ("landlord", "landlord_up", "landlord_down")
