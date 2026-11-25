---
name: pufferlib5-industrial-env
description: >-
  Designs, implements, verifies, and trains throughput-maximizing industrial
  automation environments (machines, finite buffers, changeovers, tool wear/PM,
  and re-entrant routing) as native C/CUDA environments in PufferLib 5.0. Use
  when creating, reviewing, debugging, or training a custom PufferLib 5.0 Ocean
  environment (ocean/<env>/<env>.h and binding.c) with PuffeRL (./build.sh,
  ./puffer train, ./puffer eval).
---

# PufferLib 5.0 Industrial Automation RL Environment Builder

## Overview

End-to-end engineering workflow for turning an industrial throughput control problem into a verified, high-throughput native C/CUDA environment in **PufferLib 5.0** and training/auditing a PPO policy with **PuffeRL**.

## Dependencies

- Local **PufferLib 5.0** checkout with `./build.sh` and `./puffer` CLI.
- `uv` (or Python 3 stdlib) to run `scripts/audit_env.py`.
- No external APIs or additional agent skills required.

## Quick Start

1. Read [references/mdp_checklist.md](./references/mdp_checklist.md) and lock down the factory Markov spec + pen-and-paper physical throughput ceiling with the user.
2. Implement `ocean/<env>/<env>.h` and `ocean/<env>/binding.c` starting from [references/c_env_template.md](./references/c_env_template.md).
3. Run the static audit, CPU debug build, and non-RL baselines before compiling `--cu`:

```bash
uv run skills/pufferlib5-industrial-env/scripts/audit_env.py static-check \
  --env-dir ocean/<env> --output /tmp/<env>_static.json

./build.sh <env> --debug --cpu

uv run skills/pufferlib5-industrial-env/scripts/audit_env.py record-cpu-pass \
  --env-dir ocean/<env> --output /tmp/<env>_cpu.json

# Run always-idle, random, and domain-heuristic checks, then validate bounds:
uv run skills/pufferlib5-industrial-env/scripts/audit_env.py check-baselines \
  --env-dir ocean/<env> \
  --fastest-cycle-ticks 2 \
  --horizon 1000 \
  --idle-throughput 0 \
  --random-throughput 110 \
  --heuristic-throughput 295 \
  --has-nan false \
  --obs-out-of-bounds false \
  --output /tmp/<env>_baselines.json

./build.sh <env> --cu
./puffer train <env>
./puffer eval latest
```

## Utility Scripts

All programmatic verification lives in [scripts/audit_env.py](./scripts/audit_env.py). Always write structured results to a file with `--output` and inspect exit status (exits `1` on any failure).

- **`static-check`**: Scans `<env>.h` and `binding.c` for `malloc`/`free` inside `c_step`, global `rand()`/`srand()`, identical constant RNG seeds, `OBS_DIM`/`ACT_DIM` mismatches between `<env>.h` and `binding.c`, missing `rewards`/`terminals` clearing at the start of `c_step`, and auto-reset bugs where `c_reset` wipes `rewards`/`terminals`.
  ```bash
  uv run scripts/audit_env.py static-check --env-dir ocean/<env> --output /tmp/static.json
  ```
- **`record-cpu-pass`**: Computes SHA-256 hashes of `<env>.h` and `binding.c` after `./build.sh <env> --debug --cpu` succeeds and updates `<env-dir>/.puffer_audit_passed.json`.
  ```bash
  uv run scripts/audit_env.py record-cpu-pass --env-dir ocean/<env> --output /tmp/cpu.json
  ```
- **`check-baselines`**: Computes the hard physical throughput ceiling $\lfloor \text{horizon} / \text{fastest\_cycle\_ticks} \rfloor$, verifies `idle == 0`, zero `NaN`/`Inf`, observations within bounds, `random <= ceiling`, `heuristic <= ceiling`, and `heuristic >= random`, and seals `.puffer_audit_passed.json` so the `PreToolUse` gate unlocks `--cu` and `./puffer train`.
  ```bash
  uv run scripts/audit_env.py check-baselines \
    --env-dir ocean/<env> --fastest-cycle-ticks 2 --horizon 1000 \
    --idle-throughput 0 --random-throughput 110 --heuristic-throughput 295 \
    --has-nan false --obs-out-of-bounds false --output /tmp/baselines.json
  ```

## Workflow

### 1. Specify the Industrial MDP & Physics Bounds
- Read [references/mdp_checklist.md](./references/mdp_checklist.md).
- Confirm with the user any ambiguous factory rules (e.g., whether PM interrupts an in-progress job, what happens to a part on breakdown, arrival process, buffer capacity).
- Verify the problem is non-trivial (greedy "always run" is *not* already optimal—e.g., setups/changeovers, tool wear/PM, heterogeneous machines, or re-entrant routing).
- List the exact **Markov observation vector** (normalized to `[0, 1]`, no unbounded cumulative counters), **fixed discrete actions** (with explicit `idle` and no-op invalid handling), **reward** (`+1.0` per finished exit only), **continuing horizon** (`terminal = 0` unless genuine unrecoverable fault), and **$\gamma$** matching the longest planning trade-off.
- Compute the pen-and-paper **physical throughput ceiling** per 1,000 ticks: $\lfloor 1000 / t_{\min} \rfloor$ (scaled by number of parallel exit machines if $>1$).

### 2. Implement `ocean/<env>/<env>.h` and `binding.c`
- Start from [references/c_env_template.md](./references/c_env_template.md).
- Enforce all five core C/PufferLib 5.0 invariants:
  1. Single source of truth for `OBS_DIM` and `ACT_DIM` matching `binding.c` exactly.
  2. Zero `malloc`/`free` inside `c_step`; write directly into PufferLib's preallocated flat buffers.
  3. Per-env pure-C PRNG (`xorshift32`) seeded uniquely per factory (`seed + env_id`), never global `rand()`.
  4. At the start of `c_step`, clear `rewards[0] = 0.0f`, `terminals[0] = 0.0f`, and all one-hot observation slots to `0.0f`.
  5. On true terminal events, reset only internal state + new initial observations; never overwrite `terminals[0] = 1.0f` or that tick's final reward inside `c_step`.
- Track cumulative domain KPIs (`parts_completed`, `changeovers`, `pm_count`, `breakdowns`) in the `Log` struct, never in `observations`.

### 3. Static Audit & CPU Sanitizer Build
- Run `audit_env.py static-check --env-dir ocean/<env> --output /tmp/<env>_static.json`. Fix every error before building.
- Compile with `./build.sh <env> --debug --cpu` (AddressSanitizer / debug checks enabled).
- If compilation or sanitizers report an error, fix the exact file/line bug and re-run until clean. Then run `audit_env.py record-cpu-pass`.

### 4. Non-RL Baseline Verification (on CPU Debug Build)
- Run three non-RL controllers through the debug build and collect metrics:
  1. **Always-idle**: must produce `0` throughput and `0` reward.
  2. **Random policy**: must produce zero `NaN`/`Inf`, all observations within expected bounds, and establishes the floor throughput.
  3. **Simple domain heuristic** (e.g., FIFO or setup-matching + PM threshold): must stay $\le$ the physical ceiling and beat random, establishing the engineering benchmark for PPO.
- Run `audit_env.py check-baselines` to validate all invariants and stamp `.puffer_audit_passed.json`.
- If any baseline violates the physical ceiling (e.g., `heuristic > floor(H / t_min)`), **stop immediately**—diagnose sticky rewards, double-counting, or off-by-one timers before touching CUDA.

### 5. GPU Build & PuffeRL Training
- Compile the optimized CUDA build: `./build.sh <env> --cu`.
- Launch training: `./puffer train <env>` (override hyperparameters inline via `--section.key=value`, or sweep LR/`ent_coef`/$\gamma$ with **Protein** and visualize in **Constellation**).
- Monitor `SPS`, `explained_variance`, `clipfrac`, and `entropy` using [references/pufferl_diagnostics.md](./references/pufferl_diagnostics.md).

### 6. Mandatory Policy & Exploit Audit
- Once throughput beats your heuristic with healthy curves, **do not trust dashboard curves alone**.
- Run `./puffer eval latest` and audit actual rollout traces and logged KPIs (`throughput`, `changeovers`, `pm_count`, `breakdowns`, unseen arrival rates).
- If throughput exceeds the physical ceiling or exploits a zero-tick edge case, reject the policy, fix the simulator bug, and restart from Step 3.

## Common Mistakes

- **Skipping `--debug --cpu` and jumping straight to `--cu`**: Off-by-one `OBS_DIM` mismatches silently overwrite neighboring factories in the contiguous slab and surface as mysterious `NaN` losses or flat `explained_variance`.
- **Setting `terminal = 1` on a fixed unobserved clock cutoff**: Strips future discounted value from in-progress work, feeds conflicting targets to the critic, and punishes good actions near the cutoff.
- **Adding busy bonuses or idle penalties**: Lets the agent reward-hack slow jobs or skip preventive maintenance; pure `+1.0` per finished exit with proper $\gamma$ already penalizes unnecessary idling.
- **Trusting `explained_variance = 0.90` without `./puffer eval latest`**: PPO will accurately predict and master a consistent simulator bug just as eagerly as real factory physics.
