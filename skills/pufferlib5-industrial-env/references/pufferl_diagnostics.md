# PuffeRL Dashboard Diagnostics & Policy Audit Guide

Use this reference during **Phases 5 & 6** (`./puffer train` and `./puffer eval latest`).

## 1. Dashboard Telemetry Lookup Table

| Symptom on PuffeRL Dashboard | Likely Root Cause | Exact Fix |
| :--- | :--- | :--- |
| **`SPS` unexpectedly low** | CPU/GPU sync stall, `malloc`/lock contention inside `c_step`, global `rand()` serializing OpenMP threads, or poor batch sizing. | Re-run `audit_env.py static-check`; ensure pure C per-env PRNG, zero dynamic allocation, and compiled with `--cu`. |
| **`explained_variance` stuck $\le 0.1$**, `entropy` stays flat/high, throughput stuck at random floor | Critic cannot predict returns from observations: `binding.c` dim mismatch (corrupted obs), missing Markov state (timer/setup/wear/pass), stale un-cleared one-hots, unscaled `0..50000` feature, or `terminal = 1` on an unobserved fixed cutoff. | Re-run `--debug --cpu` sanitizer build, verify observation normalization `[0, 1]`, check Markov checklist, and confirm `terminal = 0` for continuing lines. |
| **`explained_variance` healthy (`0.8+`)**, **`clipfrac` spikes high (`> 0.4`)**, **`entropy` crashes to `~0.01`** early, throughput stalls below heuristic | Oversized PPO updates driving premature convergence to a greedy local optimum. | **Lower learning rate** (bring `clipfrac` into `0.05–0.20`) and **raise `ent_coef`** so exploration decays smoothly. Or run **Protein** to sweep LR & `ent_coef` automatically. |
| **`clipfrac == 0.0`**, policy barely changes | Learning rate too low or gradient signal vanished. | Raise learning rate slightly; verify rewards are non-zero under random rollouts. |
| **Policy beats random but ignores preventive maintenance / batching** | Discount factor $\gamma$ too small for the causal delay $L$ (e.g., $0.80^{15} \approx 3.5\%$), or shaping penalties (busy bonus / idle penalty) bias against PM. | Raise $\gamma$ (e.g., `0.99`) so $\frac{1}{1-\gamma} \gg L$, and strip shaping bonuses/penalties back to pure `+1.0` per finished exit. |
| **Training loss turns `NaN`** | Out-of-bounds memory write overflowing adjacent factory buffers, divide-by-zero in normalization, or corrupted `binding.c` metadata. | Stop GPU training immediately; rebuild with `./build.sh <env> --debug --cpu` and fix the AddressSanitizer line. |

## 2. Automated Tuning & Visualization in PufferLib 5.0
- **Inline config overrides:** `./puffer train <env> --train.learning_rate=3e-4 --train.ent_coef=0.02 --train.gamma=0.99` (check your local checkout's exact config flags with `./puffer train --help`).
- **Protein:** Use PufferLib's cost-aware Bayesian optimizer to sweep LR, `ent_coef`, and $\gamma$ on the Pareto frontier of speed vs. throughput score.
- **Constellation:** Inspect multi-run sweeps and copy top hyperparameter sets.

## 3. Mandatory `./puffer eval latest` Audit
Healthy dashboard curves (`explained_variance = 0.90`, `clipfrac = 0.11`, +25% over heuristic) **never guarantee valid physics**—PPO happily learns to exploit any consistent simulator bug. Before accepting a policy:
1. Run `./puffer eval latest` and inspect episode traces + `Log` KPIs (`parts_per_window`, `changeovers`, `pm_count`, `breakdowns`).
2. Confirm evaluated throughput is **strictly $\le$ your pen-and-paper physical ceiling** $\lfloor H / t_{\min} \rfloor$.
3. Verify zero-tick transitions do not occur (e.g., PM actually consumes `PM_TICKS`, changeovers actually consume `SETUP_TICKS`).
4. Stress-test on out-of-distribution arrival rates / buffer sizes to ensure the policy didn't overfit to a single arrival seed.
5. If an exploit is found, reject the policy, fix `<env>.h`, and re-run the full verification chain (`static-check` → `--debug --cpu` → baselines → `--cu` → `./puffer train` → `./puffer eval latest`).
