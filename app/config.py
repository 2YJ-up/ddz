from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApiConfig:
    n_samples: int = 256
    rollout_per_action: int = 64
    mode: str = "perfectdou_monte_carlo"
    timeout_ms: int = 3000
