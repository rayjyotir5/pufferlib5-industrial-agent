# PuffeRL Dashboard Diagnostics & Multi-Day Policy Audit Guide

Use this reference during **Phases 5 & 6** (`./puffer train` and `./puffer eval latest`).

## 1. Dashboard Telemetry Lookup Table

| Symptom on PuffeRL Dashboard | Likely Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`SPS` unexpectedly low** | CPU/GPU sync stall, `malloc`/lock contention inside `c_step`, global `rand()` serializing OpenMP threads, or slow pathfinding inside inner loop. | Re-run `audit_env.py static-check`; use lock-free `xorshift32`, zero dynamic allocation, precomputed/O(1) routing heuristics + PIBT, and compile with `--cu`. |
| **`explained_variance` stuck $\le 0.1$**, `entropy` stays flat/high, throughput stuck at random floor | Critic cannot predict returns: `binding.c` dim mismatch (corrupted obs), missing Markov state (timer/setup/wear/battery/shift clock), stale un-cleared one-hots, unscaled `0..50000` feature, or `terminal = 1` on an unobserved fixed cutoff. | Re-run `--debug --cpu` sanitizer build, verify all obs lie in `[0, 1]`, audit Markov checklist, and keep `terminal = 0` across continuing shift boundaries. |
| **`explained_variance` healthy (`0.8+`)**, **`clipfrac` spikes high (`> 0.4`)**, **`entropy` crashes to `~0.01`** early, throughput stalls below heuristic | Oversized PPO updates driving premature convergence to a greedy local optimum. | **Lower learning rate** (bring `clipfrac` into `0.05–0.20`) and **raise `ent_coef`** so exploration decays smoothly. Or run **Protein** to sweep LR & `ent_coef`. |
| **`clipfrac == 0.0`**, policy barely changes | Learning rate too low or reward signal vanished. | Raise learning rate slightly; verify random rollouts produce non-zero reward variance. |
| **Policy beats random but ignores PM or mid-shift charging** | Discount factor $\gamma$ too small for the causal delay ($\frac{K}{1-\gamma} \ll L$), or individual (non-team) rewards discourage agents from charging. | Raise $\gamma$ (e.g., `0.998–0.999` for macro-stepped 24h shifts), use cooperative team rewards (`+1/N`), and strip shaping hacks. |
| **Policy deliberately crashes battery / triggers hard terminal early** | **Anti-suicide inequality violated:** continuous negative penalties (e.g., line stalls) accumulate worse than the terminal failure penalty ($|P_{\text{term}}| \le \frac{c_{\max}}{1 - \gamma}$). | Shrink per-step stall penalty and/or raise hard-terminal penalty magnitude until $P_{\text{term}} > \frac{c_{\max}}{1 - \gamma}$; verify with `audit_env.py check-baselines`. |
| **Training loss turns `NaN`** | Out-of-bounds memory write overflowing adjacent agent/factory slices, divide-by-zero in normalization, or corrupted `binding.c` metadata. | Stop GPU training immediately; rebuild with `./build.sh <env> --debug --cpu` and fix the AddressSanitizer line. |

## 2. Automated Tuning, Asset Sizing (Protein) & Visualization (Constellation)
- **Inline config overrides:** `./puffer train <env> --train.learning_rate=3e-4 --train.ent_coef=0.02 --train.gamma=0.998` (check exact flags in your checkout via `./puffer train --help`).
- **Protein for hyperparameters *and* optimal asset sizing:**
  - Use **Protein** (CARBS) to sweep LR, `ent_coef`, and $\gamma$ along the compute-cost vs. score Pareto frontier.
  - When **asset counts** (e.g., number of AMRs `N`, chargers, or buffer slots) are *not* fixed inputs and you need the **minimum feasible count**, expose them as runtime env config parameters (backed by static `MAX_*` C arrays) and include them in the Protein search space **jointly** with PPO hyperparameters—preventing a small `N` from failing merely due to a bad learning rate.
  - **Crucial objective guard:** Do *not* optimize raw throughput alone when sweeping `N`, or Protein will trivially pick `MAX_N`. Instead, score runs with a feasibility-gated / capex-penalized objective (e.g., reward hitting target OTIF with `stalls == 0` and `battery_deaths == 0`, minus a small cost per asset `N`) and read the exact `N` vs. OTIF Pareto frontier in **Constellation**.
- **Constellation:** Visualize large multi-run sweeps (including asset count vs. throughput/stalls/min-SOC point clouds) and copy winning hyperparameter + asset configs directly.

## 3. Mandatory `./puffer eval latest` & Multi-Day Steady-State Audit

Healthy dashboard curves (`explained_variance = 0.90`, `clipfrac = 0.11`, beating your heuristic) **never guarantee valid physics**—PPO eagerly masters consistent simulator bugs. Before accepting any policy:

1. **Physics & exploit audit:** Run `./puffer eval latest` and verify:
   - Evaluated throughput stays **within pen-and-paper cycle/demand ceilings**.
   - Minimum trip/PM/setup durations match physical kinematics (zero-tick exploits absent).
   - Zero vertex collisions, zero edge-swap violations, and zero entries into off-limits cells.
2. **Multi-day steady-state sustainability check (required for multi-shift / battery systems):**
   - Evaluate across **3+ consecutive 24-hour cycles** (e.g., 72 simulated hours).
   - Verify `min_soc`, end-of-overnight SOC, queue depths, and hourly throughput show **zero multi-day drift** (catching policies that look great on Day 1 by draining fleet energy they cannot recover during the overnight window).
3. **Out-of-distribution stress test:** Vary arrival seeds / slight demand perturbations (`±5–10%`) to confirm robustness.
4. If any exploit, collision, or multi-day drift appears, reject the policy, fix `<env>.h`, and re-run the full verification chain (`static-check` → `--debug --cpu` → baselines → `--cu` → `./puffer train` → `./puffer eval latest`).
