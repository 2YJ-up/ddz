from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.services.recommendation_service import RecommendationOptions, RecommendationService


def create_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("FastAPI is not installed. Install requirements.txt to enable /recommend.") from exc

    service = RecommendationService()
    api = FastAPI(title="Doudizhu Visible Information Decision Assistant", version="3.0")

    @api.post("/recommend")
    def recommend(payload: dict[str, Any]) -> dict[str, object]:
        raw_state = payload.get("state", payload)
        raw_options = payload.get("options", {})
        try:
            return service.recommend(raw_state, raw_options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "perfectdou_available": service.perfectdou_adapter.available}

    return api


def main() -> None:
    args = _parse_args()
    if args.serve:
        _serve(args)
        return
    if args.input is None:
        raise SystemExit("--input is required unless --serve is used")

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    raw_state = payload.get("state", payload)
    raw_options = payload.get("options", {})
    options = RecommendationOptions.from_mapping(
        {
            **raw_options,
            "n_samples": args.n_samples if args.n_samples is not None else raw_options.get("n_samples", 256),
            "rollout_per_action": (
                args.rollout_per_action
                if args.rollout_per_action is not None
                else raw_options.get("rollout_per_action", 64)
            ),
            "mode": args.mode if args.mode is not None else raw_options.get("mode", "perfectdou_monte_carlo"),
            "timeout_ms": args.timeout_ms if args.timeout_ms is not None else raw_options.get("timeout_ms", 3000),
        }
    )
    service = RecommendationService()
    if args.dry_run:
        result = service.dry_run(raw_state)
    else:
        result = service.recommend(raw_state, options)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doudizhu v3 visible-information recommendation service.")
    parser.add_argument("--input", type=Path, default=None, help="RecognizedFrameState JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and validate input without recommending.")
    parser.add_argument("--mode", default=None, help="Recommendation mode.")
    parser.add_argument("--n-samples", type=int, default=None, help="Hidden-hand sample count.")
    parser.add_argument("--rollout-per-action", type=int, default=None, help="Rollout count per action.")
    parser.add_argument("--timeout-ms", type=int, default=None, help="Timeout budget in milliseconds.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server exposing POST /recommend.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    return parser.parse_args()


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise SystemExit("uvicorn is not installed. Install requirements.txt to run --serve.") from exc
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
