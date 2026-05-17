# AGENTS.md

## Project

Build a visible-information DouDizhu decision assistant.

## Safety / Scope

Only use visible information from the user's own hand, public played cards,
public player roles, and public card counts. Do not implement hidden-card
reading, client hacking, automatic gameplay control, anti-detection, or
platform bypass logic.

## External Dependencies

PerfectDou belongs at `external/PerfectDou`. Do not modify its source. Integrate
through `src/adapters/perfectdou_adapter.py`.

## Card Encoding

Business-layer cards are strings: `3,4,5,6,7,8,9,10,J,Q,K,A,2,X,D`.
PerfectDou integer encoding is allowed only inside the PerfectDou adapter.

## Tests

Run:

```powershell
python -m pytest -q
```

## Completion Criteria

The assistant must accept `examples/midgame_state.json` and return top 3
recommended actions with score/win_rate, reasons, risks, belief_summary, and
state_warnings.
