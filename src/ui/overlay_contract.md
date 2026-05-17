# v3 Overlay Response Contract

The overlay must consume the `/recommend` response only. It must not control,
click, inject, or automate the game client.

Required display fields:

- `best_action`
- `top_actions[].action`
- `top_actions[].win_rate` when not null, otherwise `top_actions[].score`
- `top_actions[].confidence`
- `top_actions[].reasons`
- `top_actions[].risks`
- `state_warnings`
- `belief_summary`

When `win_rate` is null, label the value as score. When `state_warnings` is not
empty, show a visible low-confidence/verification notice.
