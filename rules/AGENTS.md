# PufferLib 5.0 Industrial Automation RL — Hard Invariants

Follow these rules whenever designing, editing, reviewing, compiling, or training a native C/CUDA PufferLib 5.0 environment for industrial throughput control (`ocean/<env>/`):

## 1. Markov State & Action Space
- Include every stateful variable that determines future transitions in `observations`: explicit machine mode (one-hot encoded, never a collapsed binary busy/idle flag when processing, setup, PM, and breakdown behave differently), remaining ticks in the current mode, current setup/changeover state (even when idle), wear/degradation counters, and buffer counts split by distinct remaining route stage (e.g., first-pass vs. final-pass in re-entrant lines).
- Do NOT put unbounded cumulative counters (such as `total_parts_produced_so_far`) into `observations`; put them in the `Log` struct instead.
- Normalize all continuous/count observation features to roughly `[0, 1]` (or `[-1, 1]`).
- Keep the action space fixed-size and discrete; include an explicit `idle` / `no-op` action, and treat invalid actions (e.g., loading a busy machine or pulling from an empty buffer) as safe no-ops rather than terminating the episode.
- Never invent undocumented factory physics (such as whether PM interrupts a running job or whether a breakdown scraps an in-process part); ask the user when rules are ambiguous.

## 2. Throughput Reward & Episode Horizon
- Default reward for throughput maximization is pure `+1.0f` when a finished good exits the line and `0.0f` otherwise. Avoid busy/utilization bonuses, idle penalties, or unverified intermediate move rewards that enable reward hacking; $\gamma < 1$ already applies an automatic time-delay cost to unnecessary idling.
- For continuing (24/7) factory lines where tick count is not observed, keep `terminals[i] = 0.0f` and log throughput over rolling windows. Never set `terminals[i] = 1.0f` on an arbitrary unobserved time cutoff (it corrupts critic value targets and creates false negative advantages).
- Choose discount factor $\gamma$ so the effective horizon $\frac{1}{1-\gamma}$ comfortably spans the longest operational trade-off (e.g., $\gamma \ge 0.99$ when proactive maintenance prevents a breakdown 15–30 ticks later).

## 3. PufferLib 5.0 Memory & Parallelism
- Keep `observations`, `actions`, `rewards`, and `terminals` as separate contiguous flat buffers owned by PufferLib; write directly into `env->observations[0 .. OBS_DIM - 1]`, `env->rewards[0]`, and `env->terminals[0]`.
- `OBS_DIM` and `ACT_DIM` in `binding.c` MUST exactly match the indices read/written in `<env>.h`.
- Never call `malloc`, `calloc`, `realloc`, or `free` inside `c_step`.
- Never call global libc `rand()` / `srand()` inside `c_step` or `c_reset`. Store a per-env PRNG state (`unsigned int rng` or `uint64_t rng`) inside the env struct and seed each parallel factory uniquely (e.g., `base_seed + env_id`), never with an identical constant.

## 4. Step & Auto-Reset Lifecycle
- At the **start** of every `c_step`: explicitly clear `env->rewards[0] = 0.0f`, `env->terminals[0] = 0.0f`, and zero all one-hot / sparse observation slots before writing new values.
- On a genuine terminal failure (e.g., unrecoverable deadlock or safety fault): reset only the internal factory state and populate the new episode's initial observations, while keeping `env->terminals[0] = 1.0f` and that step's final reward intact for PuffeRL to read. Never let an auto-reset overwrite `terminal` or `reward` back to `0` before `c_step` returns.

## 5. Mandatory Build, Verify & Eval Order
Never skip straight to `--cu` or `./puffer train`. Always follow this exact order after any C edit:
1. Run `audit_env.py static-check` on `ocean/<env>/`.
2. Compile with `./build.sh <env> --debug --cpu` (memory sanitizers enabled) and fix any bounds/memory errors.
3. Run three non-RL baseline checks on the debug build and verify with `audit_env.py check-baselines`:
   - **Always-idle** → `0` throughput and `0` reward.
   - **Random policy** → observations bounded, zero `NaN`/`Inf`, establishes floor throughput.
   - **Simple domain heuristic** (e.g., FIFO / setup-matching + PM threshold) → throughput stays strictly at or below the pen-and-paper physical ceiling $\lfloor H / t_{\min} \rfloor$, and sets the benchmark PPO must beat.
4. Only after steps 1–3 pass, compile with `./build.sh <env> --cu` and run `./puffer train`.
5. Never trust healthy PPO curves alone (`explained_variance` can be `0.9+` on a buggy simulator). Always audit the trained policy with `./puffer eval latest`, verifying real throughput stays below the physical ceiling and checking operational KPIs (changeovers, PM triggers, breakdowns) for simulator exploits.
