from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.core.cards import normalize_cards


def main() -> None:
    args = parse_args()
    args.file.parent.mkdir(parents=True, exist_ok=True)
    if args.clear:
        payload = {"enabled": False, "updated_at_ms": int(time.time() * 1000)}
    elif args.lead:
        ttl_seconds = 10.0 if args.ttl_seconds is None else args.ttl_seconds
        payload = {
            "enabled": True,
            "force_lead": True,
            "ttl_seconds": ttl_seconds,
            "updated_at_ms": int(time.time() * 1000),
        }
    else:
        if not args.cards:
            raise SystemExit("--cards is required unless --lead or --clear is used")
        ttl_seconds = 30.0 if args.ttl_seconds is None else args.ttl_seconds
        payload = {
            "enabled": True,
            "last_table_player": args.player,
            "last_table_cards": normalize_cards(args.cards),
            "ttl_seconds": ttl_seconds,
            "updated_at_ms": int(time.time() * 1000),
        }
    args.file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set or clear the live table-card override consumed by live_state_writer.")
    parser.add_argument("--file", type=Path, default=Path("examples/live_override.json"))
    parser.add_argument("--player", choices=["landlord", "landlord_down", "landlord_up"], default="landlord")
    parser.add_argument("--cards", nargs="*", default=[])
    parser.add_argument("--ttl-seconds", type=float, default=None)
    parser.add_argument(
        "--lead",
        "--new-trick",
        "--free-turn",
        action="store_true",
        help="Force the next recommendation to treat the position as a free lead with no previous hand.",
    )
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
