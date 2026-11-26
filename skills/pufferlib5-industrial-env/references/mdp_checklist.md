# Industrial MDP, Conservation Audit & Reward Safety Checklist

Use this reference during **Phases 0 & 1** before writing any C code.

## 1. Phase 0: Pre-Design Conservation & Feasibility Audit

Before defining observations or actions, run three pen-and-paper checks:
1. **Non-triviality check:** Confirm greedy "always run" is *not* already optimal (e.g., changeovers/setups, tool wear vs. PM, re-entrant routing, finite buffer blocking, or multi-AMR fleet + charger scheduling).
2. **Flow & spatial conservation:** Sum inflows vs. outflows per physical zone. Any net imbalance forces unavoidable empty deadhead trips that must be included in capacity estimates.
3. **Multi-day energy / resource conservation:** Compare total energy/resource burned per operating day against off-shift replenishment capacity (e.g., total fleet battery-hours burned vs. `num_chargers × off_shift_hours`).
   - If daily burn exceeds off-shift replenishment, **in-shift replenishment is physically mandatory** and you **must** simulate continuous multi-day cycles (a single shift starting at 100% SOC lets policies cheat by ending the shift empty).
4. **Outer config sweep vs. inner RL action (using Protein for asset sizing):** Structural sizing choices (minimum fleet size `N`, buffer capacities, number of chargers) belong as runtime environment config parameters—not RL actions. When optimal asset counts are unknown, size internal C arrays with static `MAX_*` caps and let **Protein** sweep asset counts jointly with PPO hyperparameters (LR, `ent_coef`, $\gamma$) against a capex-penalized OTIF objective, inspecting the `N` vs. OTIF Pareto frontier in **Constellation**.

## 2. Mandatory Phase 1 Intake Questionnaire

Resolve these six questions with the user before writing C structs (never guess silent physics):
1. **Objective shape:** Maximize uncapped throughput, or hit a fixed target rate OTIF with minimum resources (`N`)?
2. **Station binding & off-limits geometry:** Are tasks bound to specific pickup/drop stations or poolable across a zone? Are any map vertices unused or off-limits (no entry / no parking)?
3. **Arrivals, full buffers & drops:** Deterministic or stochastic (Poisson) arrivals? When a pickup buffer hits capacity, are excess arrivals **lost** or does the upstream line **stall** (pausing arrivals)? Do drop stations absorb immediately or can drops also back up?
4. **Multi-phase / shift boundaries:** Single shift or continuous multi-day cycle (e.g., 16h active + 8h off-shift)? Should the same policy and action space run across all phases, and do pending/in-flight tasks finish draining after shift end?
5. **Resource drain & hard failure rules:** Does battery/wear drain strictly on motion/processing ticks or also during idle/turns/dwells? What happens at `0%` (hard episode terminal, stall, or tow)?
6. **Idle behavior & traffic:** Can idle agents park anywhere, in designated staging zones, or must the policy learn where to park under physical traffic yielding?

## 3. Timescale & Stepping Architecture Matrix

| Problem Profile | Recommended Stepping Pattern | Why |
| :--- | :--- | :--- |
| Short-horizon single cell (cycle ~2–10 ticks, PM horizon ~15–50 ticks) | **1-tick single-agent** (`num_agents = 1`, `K = 1`) | $\gamma = 0.99$ (`100`-tick horizon) spans all trade-offs directly. |
| Multi-agent fleet / multi-hour shift & charging (1s moves, 75s trips, 4h charging, 24h cycle) | **Lockstep Multi-Agent Macro-Step** (`num_agents = N`, `K = 15..60` physical ticks per `c_step` + next-task pre-assignment + interruptible charging/parking) over a deterministic 1-tick C traffic kernel (e.g., PIBT) | Native PufferLib 5.0 lockstep arrays, zero wasted idle gap between drops, continuous partial charging with one `charge` action, and $\frac{K}{1-\gamma}$ spans hours with $\gamma \approx 0.998\text{–}0.999$. |
| Irregular long asynchronous tasks with small `N` | **Event-driven turn-taking** (`num_agents = 1`, `c_step` advances until next free agent) | Exact 1-tick wakeups, though interleaved trajectories make credit assignment noisier and require discrete charge-duration buckets. |

**Mandatory $\gamma$ horizon check:** Verify that the effective planning horizon in physical ticks, $H_{\text{eff}} = \frac{K}{1 - \gamma}$, comfortably exceeds the longest causal delay $L$ (maintenance lead time, charging duration, or shift-to-shift recovery).

## 4. Markov Observation Completeness Audit

Check every stateful variable so a memoryless (MLP) policy never faces hidden state:

| Component | Required Normalized `[0, 1]` Fields | Common Omissions |
| :--- | :--- | :--- |
| **Each machine / agent** | Explicit mode **one-hot** (`idle/proc/setup/pm/broken` or `idle/to_pick/to_drop/charging/parking`), position `(x, y)` if mobile, remaining ticks in current mode/trip, current setup/SKU class (persisted even while idle!), wear or battery SOC, pre-assignment flag. | Collapsing busy modes into binary `0/1`; omitting `remaining_timer`, `current_setup_type`, or `wear`/`battery`. |
| **Buffers & flows** | Per-station buffer fill `buf[s] / BUF_CAP` and `is_stalled[s]`; counts split by distinct **remaining route stage** (e.g., first-pass vs. final-pass in re-entrant lines); distance from deciding agent to oldest pickup and `amrs_assigned_to_flow / N` (prevents herding). | Merging re-entrant passes; adding redundant `empty_slots`; omitting in-flight peer counts so multiple AMRs dogpile the same station. |
| **Shared resources & clock** | Per-charger occupancy & occupant SOC, fleet zone fractions, and **cycle clock** (`is_active_shift`, `phase_time_remaining`) for multi-shift systems. | Omitting shift clock so policy charges unnecessarily right before shift end or fails to pre-charge for overnight recovery. |
| **KPI hygiene** | Keep all cumulative counters (`total_parts`, `stalled_mins`, `battery_deaths`) strictly in `Log`, **never** in `observations`. | Letting unbounded `0..50000` counters into obs, dragging the MLP out of distribution. |

## 5. Fixed Discrete Action Space Design

- Fixed action count every step; always include an explicit `idle` / `park` option (and staging zones for mobile fleets).
- Treat invalid actions (busy machine, empty flow queue, full chargers) as safe no-ops—never terminate the episode on an invalid action.
- With macro-stepping (`K` ticks), a single interruptible `charge` action enables arbitrary partial charging (e.g., 30% → 67%) without hand-tuned SOC buckets.

## 6. Throughput Reward, Multi-Agent Alignment & Anti-Suicide Math

- **Pure throughput reward:** `+1.0f` per finished exit (`0.0f` otherwise). Never reward raw takt time (time/part—maximizing it stalls the line), and avoid busy bonuses or arbitrary idle penalties ($\gamma < 1$ already penalizes delay).
- **Multi-flow cherry-picking guard:** When routes have unequal trip lengths (e.g., 14 vs. 72 ticks), ensure rate-capped arrivals plus finite pickup buffer caps (blocking/stalling the line at `BUF_CAP`) prevent farming short hops at the expense of long routes.
- **Cooperative team reward (`num_agents = N`):** Distribute `+1.0f / N` to every agent on any fleet delivery so agents willingly accept long routes or rotate to chargers for team throughput.
- **Anti-suicide inequality (mandatory when mixing continuous negative penalties + hard terminal failure):**
  If an agent can incur up to $c_{\max} > 0$ in continuous negative penalties per macro-step (e.g., stalled pickup stations) while a hard failure (`terminals = 1.0f`, e.g., 0% battery) applies penalty magnitude $P_{\text{term}} > 0$, enforce:
  $$P_{\text{term}} > \frac{c_{\max}}{1 - \gamma}$$
  Otherwise, an early policy can achieve higher return by deliberately killing its battery to escape ongoing stall penalties.

## 7. Pen-and-Paper Physical & Demand Ceilings

Compute the applicable upper bound(s) before running any code:
- **Single-cell / uncapped cycle ceiling:** $\text{MaxThroughput}(H) = \lfloor H / t_{\min} \rfloor \times N_{\text{exit\_machines}}$.
- **Rate-capped stochastic demand ceiling:** Mean arrival rate $\Lambda$ (e.g., `330 pallets/hr`) over long windows (allowing modest Poisson variance on finite samples), plus minimum trip time $\ge \text{Manhattan} + \text{turns} + \text{dwells}$ and **zero** collisions / off-limits intrusions.
