# PufferLib 5.0 Industrial Automation RL — Hard Invariants

Follow these rules whenever designing, editing, reviewing, compiling, or training a native C/CUDA PufferLib 5.0 environment for industrial throughput control (`ocean/<env>/`):

## 1. Pre-Design Conservation Audit & Intake (Before Writing Any C)
- Run pen-and-paper **conservation checks** before choosing RL architecture:
  - **Flow / spatial balance:** compute inflows vs. outflows per zone to identify forced deadhead.
  - **Multi-day energy / resource balance:** compare resource consumption per cycle (e.g., battery movement steps, tool wear) against replenishment capacity (e.g., overnight charger-hours). If overnight replenishment cannot cover daily consumption, you **must** model multi-day continuous operation rather than single shifts starting at 100%.
  - **Outer sweep vs. inner RL action:** treat fleet size `N`, buffer capacities, and charger counts as environment config parameters to sweep—not RL actions. When asset counts are unknown and must be minimized, expose them as runtime config knobs (backed by fixed `MAX_*` array caps in C) and use PufferLib 5.0's **Protein** to sweep asset counts jointly with PPO hyperparameters (LR, `ent_coef`, $\gamma$), scoring against a capex-penalized OTIF objective and inspecting the asset-count Pareto frontier in **Constellation**.
- Ask the user to resolve ambiguous factory rules before coding (objective shape, station vs. zone task binding, off-limits cells, deterministic vs. stochastic arrivals, full-buffer stall vs. drop behavior, drop-station absorption, multi-shift continuity, idle/turn/dwell resource drain, and 0%-resource failure rules). Never invent undocumented factory physics.

## 2. Timescale, Markov State & Action Space
- Match stepping granularity to the longest causal trade-off:
  - Use **1-tick single-agent stepping** only for short-horizon cells.
  - When dynamics span orders of magnitude (e.g., 1-second grid motion vs. 4-hour charging across a 24-hour cycle), use **lockstep multi-agent macro-stepping (`num_agents = N`, `K` physical ticks per `c_step` + next-task pre-assignment + interruptible charging/parking)** over a deterministic 1-tick C physics/collision kernel (e.g., PIBT).
  - Verify $\frac{K}{1 - \gamma}$ comfortably spans the longest causal trade-off (maintenance, charging, or Day 1 → Day 2 recovery).
- Include every stateful variable that determines future transitions in `observations`: explicit mode one-hot, remaining ticks in current mode, current setup/changeover state (even when idle), wear/battery levels, per-station buffer fill & stall flags, buffer counts split by distinct remaining route stage, and phase/shift clock for multi-phase cycles.
- Do NOT put unbounded cumulative counters (such as `total_parts_produced_so_far`) into `observations`; put them in the `Log` struct instead.
- Normalize all continuous/count observation features to roughly `[0, 1]` (or `[-1, 1]`).
- Keep the action space fixed-size and discrete; include an explicit `idle` / `park` action, and treat invalid actions as safe no-ops rather than terminating the episode.

## 3. Throughput Reward, Multi-Agent Alignment & Anti-Suicide Math
- Default throughput reward is pure `+1.0f` per finished exit (`0.0f` otherwise). Avoid busy bonuses or arbitrary idle penalties—$\gamma < 1$ already discounts unnecessary delay.
- **Multi-flow cherry-picking guard:** When routes have vastly different cycle times, confirm that rate-capped arrivals plus finite pickup stall caps (or flow SLAs) prevent the policy from farming short routes while starving long ones.
- **Cooperative multi-agent fleets (`num_agents = N`):** Use shared team throughput reward (`+1.0f / N` per delivery across all agents) so agents willingly take long hauls or rotate to chargers for fleet throughput instead of fighting over short jobs.
- **Anti-suicide inequality:** Whenever an env combines a **continuous negative penalty** (e.g., stalled-line penalty $c_{\max}$ per agent per step) with a **hard terminal failure** (e.g., 0%-battery death with penalty magnitude $P_{\text{term}}$ and `terminals = 1.0f`), enforce $P_{\text{term}} > \frac{c_{\max}}{1 - \gamma}$ so deliberately triggering terminal failure can never beat staying alive.
- For continuing (24/7 or multi-day) lines where episode length is arbitrary, keep `terminals[a] = 0.0f` across cycle boundaries and reserve `terminals[a] = 1.0f` strictly for genuine hard failures.

## 4. PufferLib 5.0 Memory, Parallelism & Auto-Reset Lifecycle
- Keep `observations`, `actions`, `rewards`, and `terminals` as separate contiguous flat buffers owned by PufferLib; write directly into each agent's slice (`obs_dim` per agent).
- `OBS_DIM` and `ACT_DIM` in `binding.c` MUST exactly match the indices read/written in `<env>.h`.
- Never call `malloc`, `calloc`, `realloc`, or `free` inside `c_step`.
- Never call global libc `rand()` / `srand()`. Store a per-env pure-C PRNG (`xorshift32`) inside the env struct and seed each parallel instance uniquely (`base_seed + env_id`).
- At the **start** of every `c_step`: explicitly clear `rewards[a] = 0.0f`, `terminals[a] = 0.0f`, and all one-hot/sparse observation slots for every agent `a` before writing new values.
- On a genuine terminal failure: reset only internal state and new initial observations while preserving `terminals[a] = 1.0f` and that step's final reward until `c_step` returns.

## 5. Mandatory Build, Verify & Multi-Day Eval Order
Never skip straight to `--cu` or `./puffer train`. Always follow this exact order after any C edit:
1. Run `audit_env.py static-check` on `ocean/<env>/`.
2. Compile with `./build.sh <env> --debug --cpu` (memory sanitizers enabled), fix any bounds/memory errors, and run `audit_env.py record-cpu-pass`.
3. Run three non-RL baseline checks on the debug build and verify with `audit_env.py check-baselines` (including demand/cycle ceilings, zero collisions/off-limits intrusions, and anti-suicide bounds):
   - **Always-idle** → `0` throughput and `0` delivery reward.
   - **Random policy** → observations bounded, zero `NaN`/`Inf`, establishes floor throughput.
   - **Simple domain heuristic** → throughput stays within physical/demand bounds, verifies zero deadlocks/collisions, and sets the benchmark PPO must beat (sweeping `N` if sizing a fleet).
4. Only after steps 1–3 pass, compile with `./build.sh <env> --cu` and run `./puffer train`.
5. Never trust healthy PPO curves alone (`explained_variance` can be `0.9+` on a buggy simulator). Always audit with `./puffer eval latest`—and for continuous multi-shift systems, run **3+ consecutive 24-hour cycles** to verify zero multi-day SOC/queue drift, zero collisions, and zero physics exploits.
