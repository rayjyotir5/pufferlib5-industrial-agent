# Industrial MDP, Reward & Physics Ceiling Checklist

Use this reference during **Phase 1** before writing any C code.

## 1. Non-Triviality Check
A single-product serial line (`A -> Buffer -> B`) with no setups, failures, or routing choices is solved optimally by a greedy "always run" rule. Confirm at least one genuine operational trade-off exists before investing in RL:
- **Changeovers / setups**: switching between SKU classes incurs setup ticks.
- **Tool wear & preventive maintenance (PM)**: running past a wear threshold causes a long breakdown vs. a short proactive PM.
- **Re-entrant routing**: parts revisit machines across stages (e.g., `Type 0: A -> B`, `Type 1: B -> A -> B`) sharing finite WIP buffers, creating blocking/deadlock risks.
- **Heterogeneous parallel machines**: fast/expensive vs. slow/reliable stations.

## 2. Markov Observation Completeness Audit
Check every component of the cell so a memoryless (MLP) policy can predict the exact next transition without hidden state:

| Component | Required State Fields | Common Omissions |
| :--- | :--- | :--- |
| **Each machine** | Explicit mode **one-hot** (`idle`, `processing`, `setup`, `pm`, `broken`), remaining ticks in current mode, current setup/SKU class (persisted even when idle!), current job stage/pass inside machine (if exit vs. re-entry differs), wear/degradation counter. | Collapsing busy modes into a single `0/1` bit; omitting `remaining_timer`; forgetting `current_setup_type` while idle; forgetting `wear` counters. |
| **Each finite buffer** | Counts per **remaining route stage** (e.g., `type1_first_B` separate from `type1_final_B`). Prefer class counts over raw slot arrays when slots are interchangeable to remove permutation symmetry. | Merging first-pass and second-pass re-entrant parts; adding redundant `empty_slots = CAP - sum(counts)` when capacity is fixed. |
| **Scaling & KPI hygiene** | Normalize all numeric features to roughly `[0, 1]` (e.g., `timer / MAX_TIMER`, `wear / MAX_WEAR`, `count / BUF_CAP`). | Leaving raw `0..50000` values next to `0/1` flags; putting cumulative KPIs (`total_parts_produced`) into `observations` instead of `Log`. |

## 3. Fixed Discrete Action Space Design
- Keep a fixed action set every tick (e.g., `0 = idle`, `1 = load_fast`, `2 = load_slow`, `3 = start_pm`).
- Include an explicit `idle` action so the policy can strategically wait (e.g., to avoid a changeover or reserve a buffer slot) even when the buffer is full.
- Handle invalid actions (loading a busy/broken machine, pulling from an empty queue) as **safe no-ops**—never terminate the episode on an invalid action, or random early policies die on ticks 1–2 before ever completing a part.
- Auto-unload finished parts unless holding a finished part inside a machine is an intentional blocking mechanism.

## 4. Throughput Reward & Horizon Design
- **Pure throughput reward**: `+1.0f` when a finished good exits the cell, `0.0f` otherwise.
  - Never reward raw `finished_takt_time / incoming_takt_time` (takt time is time/part—maximizing it teaches the policy to stall!).
  - Avoid busy bonuses (`+0.5/tick` while running) or explicit idle penalties (`-0.05/tick`): they incentivize slow jobs, discourage PM, or fight strategic batching. With $\gamma < 1$, every unnecessary idle tick already discounts all future `+1.0` exits automatically.
- **Continuing horizon (`terminals[0] = 0.0f`)**:
  - For 24/7 lines where tick count is not observed, keep `terminals[0] = 0.0f` so PPO always bootstraps $y_t = r_t + \gamma V(s_{t+1})$, and log rolling-window throughput (e.g., every 1,000 ticks) in `Log`.
  - Reserve `terminals[0] = 1.0f` strictly for genuine unrecoverable failures (e.g., safety fault / deadlock) or finite shifts whose remaining time is observed (or much longer than $\frac{1}{1-\gamma}$).
- **Discount factor $\gamma$**:
  - Check $\gamma^L$ over the longest causal delay $L$ (e.g., $L = 15$ ticks between a 3-tick PM decision and a 10-tick breakdown):
    - $\gamma = 0.80 \implies 0.80^{15} \approx 0.035$ (policy ignores PM and crashes).
    - $\gamma = 0.99 \implies 0.99^{15} \approx 0.86$ (effective horizon $\frac{1}{1-\gamma} = 100$ ticks; policy reliably learns proactive PM).

## 5. Pen-and-Paper Physical Throughput Ceiling
Before running any code, compute the hard upper bound over evaluation window $H$ (e.g., $H = 1000$ ticks):
$$\text{MaxThroughput}(H) = \left\lfloor \frac{H}{t_{\min}} \right\rfloor \times N_{\text{exit\_machines}}$$
where $t_{\min}$ is the shortest possible cycle time of any job at the exit station (ignoring setups, PM, and breakdowns). Any baseline or trained policy exceeding this number **proves a simulator bug** (sticky reward, double-counting, or off-by-one timer).
