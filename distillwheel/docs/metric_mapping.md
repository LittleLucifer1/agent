# Metric field mapping

Every backend's `LogParser` emits a normalized `Metric` object. This
table is the source of truth for how framework-native metric keys map
onto the unified schema.

| Unified field   | swift key                     | verl key                          | Notes |
|-----------------|-------------------------------|-----------------------------------|-------|
| `step`          | `step` / `global_step`        | `step` / `global_step` / `iteration` | required — parser drops the line if missing |
| `loss`          | `loss`                        | `actor/loss` (fallback: `loss`)   | for SFT this is CE; for RL it's the actor loss |
| `learning_rate` | `learning_rate` / `lr`        | `actor/lr` / `learning_rate` / `lr` | |
| `grad_norm`     | `grad_norm` / `grad_norm_clipped` | `actor/grad_norm` / `grad_norm` | post-clip if both are present |
| `epoch`         | `epoch`                       | `epoch`                            | fractional |
| `kl`            | `kl` / `kl_div`               | `actor/kl` / `approx_kl` / `kl`    | KL to reference policy |
| `reward_mean`   | `reward` / `reward_mean`      | `reward/mean` / `reward_mean`      | mean per-rollout reward |
| `reward_std`    | —                             | `reward/std` / `reward_std`        | swift usually doesn't emit this |
| `policy_loss`   | —                             | `actor/pg_loss` / `actor/policy_loss` | |
| `value_loss`    | —                             | `critic/loss` / `value_loss`       | PPO only |
| `entropy`       | —                             | `actor/entropy` / `entropy`        | |
| `extra`         | any other key/value           | any other key/value                | escape hatch — never break the schema for a private field |

## Rules for adding a new backend

1. Emit the same unified keys — do not invent new columns for concepts
   that already exist. If your framework uses "entropy_bonus", map it
   to `entropy`.
2. Anything genuinely new goes under `extra` as a plain dict.
3. If a line cannot be parsed, return `None` from `parse_line` — the
   pipeline drops it silently. Never fabricate a metric.
4. `stage` is set by the orchestrator from `recipe.stage` before
   parsing starts; parsers should not overwrite it.
