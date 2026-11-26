---
name: pufferlib5-industrial-env
description: >-
  Designs, implements, verifies, and trains throughput-maximizing industrial
  automation environments (single cells, re-entrant lines, and multi-agent AMR
  fleets with finite buffers, changeovers, wear/PM, charging, and multi-day
  cycles) as native C/CUDA environments in PufferLib 5.0. Use when creating,
  reviewing, debugging, or training a custom PufferLib 5.0 Ocean environment
  (ocean/<env>/<env>.h and binding.c) with PuffeRL (./build.sh, ./puffer train,
  ./puffer eval).
---

# PufferLib 5.0 Industrial Automation RL Environment Builder

## Overview

End-to-end engineering workflow for turning an industrial throughput or fleet-coordination problem into a verified, high-throughput native C/CUDA environment in **PufferLib 5.0** and training/auditing a PPO policy with **PuffeRL**.

## Dependencies

- Local **PufferLib 5.0** checkout with `./build.sh` and `./puffer` CLI.
- `uv` (or Python 3 stdlib) to run `scripts/audit_env.py`.
- No external APIs or additional agent skills required.

## Quick Start

1. Read [references/mdp_checklist.md](./references/mdp_checklist.md): run the **Phase 0 conservation audit** (flow balance, energy/replenishment balance, outer parameter sweep vs. RL action) and complete the **Phase 1 intake questionnaire** with the user before writing C.
2. Implement `ocean/<env>/<env>.h` and `ocean/<env>/binding.c` starting from [references/c_env_template.md](./references/c_env_template.md) (Single-Agent 1-tick or Multi-Agent Lockstep Macro-Step).
3. Run the static audit, CPU debug build, and non-RL baselines before compiling `--cu`:

```bash
uv run skills/pufferlib5-industrial-env/scripts/audit_env.py static-check \
  --env-dir ocean/<env> --output /tmp/<env>_static.json

./build.sh <env> --debug --cpu

uv run skills/pufferlib5-industrial-env/scripts/audit_env.py record-cpu-pass \
  --env-dir ocean/<env> --output /tmp/<env>_cpu.json

# Single-cell cycle-ceiling check OR rate-capped multi-agent fleet check:
uv run skills/pufferlib5-industrial-env/scripts/audit_env.py check-baselines \
  --env-dir ocean/<env> \
  --demand-ceiling 330 \
  --idle-throughput 0 \
  --random-throughput 95 \
  --heuristic-throughput 325 \
  --has-nan false \
  --obs-out-of-bounds false \
  --collisions 0 \
  --off-limits-intrusions 0 \
  --max-step-penalty 0.07 \
  --terminal-penalty 100.0 \
  --gamma 0.998 \
  --output /tmp/<env>_baselines.json

./build.sh <env> --cu
./puffer train <env>
./puffer eval latest
```

## Utility Scripts

All programmatic verification lives in [scripts/audit_env.py](./scripts/audit_env.py). Always write structured results to a file with `--output` and inspect exit status (exits `1` on any failure).

- **`static-check`**: Scans `<env>.h` and `binding.c` for `malloc`/`free` inside `c_step`, global `rand()`/`srand()`, identical constant RNG seeds, `OBS_DIM`/`ACT_DIM` mismatches between `<env>.h` and `binding.c`, missing `rewards`/`terminals` clearing at the start of `c_step`, and auto-reset bugs where `c_reset` wipes `rewards`/`terminals`.
- **`record-cpu-pass`**: Computes SHA-256 hashes of `<env>.h` and `binding.c` after `./build.sh <env> --debug --cpu` succeeds and updates `<env-dir>/.puffer_audit_passed.json`.
- **`check-baselines`**: Validates non-RL baselines against single-machine physical ceilings (`--fastest-cycle-ticks` + `--horizon`) and/or stochastic demand ceilings (`--demand-ceiling`, with optional `--demand-tolerance-pct`), checks `idle == 0`, zero `NaN`/`Inf`, bounded obs, `--collisions 0`, `--off-limits-intrusions 0`, and optional anti-suicide reward inequality (`--max-step-penalty`, `--terminal-penalty`, `--gamma`), then seals `.puffer_audit_passed.json` to unlock `--cu` and `./puffer train`.

## Workflow

### 0. Pre-Design Conservation & Feasibility Audit (Before Choosing RL)
- Read Section 1 of [references/mdp_checklist.md](./references/mdp_checklist.md).
- Compute **flow / spatial balance** (inflows vs. outflows per zone → unavoidable deadhead).
- Compute **multi-day energy / resource conservation** (daily consumption vs. off-shift replenishment capacity → proves whether in-shift charging and multi-day rollouts are mandatory).
- Separate **outer engineering sweeps** (fleet size `N`, buffer caps, charger counts) from **inner RL actions**.

### 1. Structured Intake & MDP / Stepping Architecture
- Run the **6-question intake** from Section 2 of [references/mdp_checklist.md](./references/mdp_checklist.md) to resolve every physics ambiguity with the user *before* coding:
  1. Objective shape (uncapped throughput vs. fixed-rate OTIF with minimum resource `N`).
  2. Station vs. zone task binding, plus any unused/off-limits geometry.
  3. Arrival process (deterministic vs. stochastic), full-buffer behavior (lost arrival vs. stalled upstream line), and drop-station absorption.
  4. Single shift vs. continuous multi-day cycle, off-shift policy behavior, and shift-end drain rules.
  5. Resource drain conditions (motion-only vs. idle/turn/dwell) and 0%-resource failure behavior.
  6. Idle/parking rules and traffic conflict resolution.
- Choose the right **stepping architecture** (1-tick single-agent vs. **Lockstep Multi-Agent Macro-Step** `num_agents = N`, `K`-tick inner C physics with next-task pre-assignment & interruptible charging/parking) and verify $\frac{K}{1 - \gamma}$ spans the longest causal trade-off.
- Lock down the exact normalized `[0, 1]` Markov `OBS_DIM`, fixed `ACT_DIM`, cooperative reward scaling, anti-suicide bound, and pen-and-paper throughput ceilings.

### 2. Implement `ocean/<env>/<env>.h` and `binding.c`
- Start from [references/c_env_template.md](./references/c_env_template.md) (Pattern A for single-machine cells, Pattern B for multi-agent lockstep macro-step fleets with deterministic PIBT collision resolution).
- Enforce all C/PufferLib 5.0 invariants: shared `OBS_DIM`/`ACT_DIM` macros, zero `malloc`/`free` in `c_step`, per-env uniquely seeded `xorshift32` PRNG, explicit clearing of `rewards`/`terminals`/one-hot obs at the start of `c_step`, safe internal auto-reset preserving `terminals = 1.0f` and final reward, and operational KPIs kept strictly in `Log`.

### 3. Static Audit & CPU Sanitizer Build
- Run `audit_env.py static-check --env-dir ocean/<env> --output /tmp/<env>_static.json`. Fix every error before compiling.
- Compile with `./build.sh <env> --debug --cpu` (AddressSanitizer enabled). Fix any reported memory/bounds error, then run `audit_env.py record-cpu-pass`.

### 4. Non-RL Baseline Verification (on CPU Debug Build)
- Run **always-idle** (`0` throughput), **random policy** (bounded obs, zero `NaN`/`Inf`, terminal reset check), and a **simple domain heuristic** (over 3+ consecutive 24h days for continuous multi-shift systems, sweeping `N` if sizing a fleet).
- Run `audit_env.py check-baselines` (passing `--fastest-cycle-ticks`/`--horizon` and/or `--demand-ceiling`, `--collisions`, `--off-limits-intrusions`, and anti-suicide parameters when applicable).
- If any baseline violates physical/demand bounds or produces collisions/deadlocks, **stop and fix `<env>.h` before touching CUDA**.

### 5. GPU Build & PuffeRL Training
- Compile `./build.sh <env> --cu` and run `./puffer train <env>` (for fleet sizing, start at a comfortable `N` first, then step `N` down).
- Monitor `SPS`, `explained_variance`, `clipfrac`, and `entropy` using [references/pufferl_diagnostics.md](./references/pufferl_diagnostics.md). Use **Protein** for hyperparameter sweeps and **Constellation** for visualizing runs.

### 6. Mandatory Policy & Multi-Day Steady-State Audit
- Never trust healthy dashboard curves alone. Run `./puffer eval latest` and audit real traces and `Log` KPIs (`delivered_per_hr`, `stalled_station_mins`, `battery_deaths`, `min_soc`, `collisions`).
- For continuous multi-shift systems, verify **3+ consecutive 24-hour cycles** with zero downward drift in `min_soc` or upward drift in queues. If any exploit or drift appears, reject the policy, fix `<env>.h`, and restart from Step 3.

## Common Mistakes

- **Skipping conservation math before coding**: Leads to single-shift episodes that "cheat" overnight energy budgets or infeasible fleet sizes.
- **Using 1-tick RL on multi-hour charging/shift horizons**: Standard PPO discounting ($\gamma = 0.99$) cannot bridge 1-second moves to 4-hour charging without macro-stepping (`K`-tick stride).
- **Mixing continuous negative penalties with hard terminal failure without checking the anti-suicide bound**: Early policies deliberately kill their battery (`terminal = 1`) to escape ongoing stall penalties.
- **Skipping `--debug --cpu` or trusting `explained_variance = 0.90` without `./puffer eval latest`**: Silent slab overflows cause `NaN`s on `--cu`, and PPO happily masters consistent simulator bugs.
