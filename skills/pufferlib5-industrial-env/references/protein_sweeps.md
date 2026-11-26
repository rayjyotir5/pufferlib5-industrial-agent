# Protein Sweeps & Optimal Asset Sizing Guide (PufferLib 5.0)

**Protein** is PufferLib's streamlined, hardened adaptation of Imbue AI's **CARBS** (*Cost-Aware Bayesian Search*). Instead of grid or random search, Protein fits a Gaussian Process over both **performance (`score`)** and **compute cost (wall-clock time / training steps)** to map the **Pareto frontier** of fast, high-scoring configurations. Pair it with **Constellation** to visualize multi-run sweeps and copy winning configs.

---

## 1. When to Use Protein — and When NOT to Use It

### Do NOT run Protein yet if:
- Your environment has not passed `audit_env.py static-check`, `./build.sh <env> --debug --cpu`, and `audit_env.py check-baselines` (`pre_train_gate.py` blocks `./puffer sweep` until these pass).
- You have not completed **at least one clean manual `./puffer train` run** with non-`NaN` loss, healthy `SPS`, and `explained_variance > 0.3`. Sweeping a broken observation space or buggy simulator only burns GPU time optimizing noise.

### DO use Protein when:
1. **PPO hyperparameter tuning:** Your single baseline run learns (`explained_variance` climbs) but plateaus below your domain heuristic or shows `clipfrac`/`entropy` sensitivity (`learning_rate`, `ent_coef`, `gamma`, `batch_size`, `bptt_horizon`).
2. **Optimal asset sizing (when asset counts are NOT fixed):** You need the **minimum** fleet size (`num_agents`), charger count, or buffer capacity that hits target throughput OTIF—sweeping asset counts **jointly** with PPO hyperparameters so a tight fleet (e.g., `N = 9`) isn't falsely dismissed due to an untuned learning rate.
3. **Bounded reward-penalty tuning:** Tuning small secondary weights (e.g., a stalled-line penalty) while strictly respecting the anti-suicide inequality $P_{\text{term}} > \frac{c_{\max}}{1 - \gamma}$.

---

## 2. Choosing Parameter Distributions (`distribution`, `min`, `max`, `mean`, `scale`)

Protein's Gaussian Process performs best when each search dimension uses the right space and is centered (`mean`) on a **verified working baseline** rather than an arbitrary guess. Keep sweeps focused on **4–7 high-leverage parameters**.

| Parameter Type | Distribution | Typical Industrial RL Examples |
| :--- | :--- | :--- |
| Positive scalars spanning orders of magnitude | `log_normal` | `learning_rate` (`1e-5..3e-3`), `ent_coef` (`1e-4..0.1`), `vf_coef` |
| Parameters bounded near `0` or `1` | `logit_normal` | `gamma` (`0.95..0.9995`), `gae_lambda` (`0.8..0.99`), `clip_coef` (`0.05..0.4`) |
| Hardware-friendly powers of 2 | `uniform_pow2` | `batch_size`, `minibatch_size`, `hidden_size` |
| Discrete integer asset/sizing counts | `int_uniform` | `num_agents` (`8..14`), `num_chargers` (`2..6`), `buf_cap` (`3..8`) |

**Sizing tip for `scale`:** Start with moderate `scale` around your working baseline `mean` so early trials stay in the learnable regime instead of jumping straight into unstable corners.

---

## 3. Sweeping Asset Counts Without Falling Into the `MAX_N` Trap

If you sweep `num_agents ∈ [8, 14]` and tell Protein to maximize raw `delivered_per_hr`, Protein will trivially pick `num_agents = 14` (more assets = easier throughput).

### Required C & Metric Design for Asset Sizing
1. **Runtime-configurable asset counts (zero recompile between trials):** Allocate all arrays statically up to compile-time caps (`#define MAX_AGENTS 16`, `#define MAX_CHARGERS 8`) inside `<env>.h` and read `num_agents` / `num_chargers` from runtime config (`--env.num_agents=N`).
2. **Capex-penalized OTIF `sweep_score` in `Log`:** Expose a dedicated scalar `log.sweep_score` that rewards meeting 100% OTIF with zero failures while penalizing extra assets:
   ```c
   // Example for target = 330 pallets/hr, alpha = 8.0 pallets/hr equivalent per AMR:
   // Feasible (0 stalls, 0 battery deaths, >= 98% target) beats any infeasible run,
   // and among feasible fleets, fewer AMRs strictly wins.
   float capex_cost = 8.0f * (float)env->num_agents;
   float infeasibility = 25.0f * stalled_station_mins_per_hr + 200.0f * battery_deaths;
   float sweep_score = delivered_per_hr - infeasibility - capex_cost;
   ```
   - Check the margin: saving 1 AMR gains `+8.0`, which strictly outweighs small stochastic variance once OTIF is met, while stalling a line (`-25.0/min`) or crashing a battery (`-200.0`) immediately pushes under-sized fleets far below feasible ones.

---

## 4. Running a Protein Sweep (`./puffer sweep`)

Inspect your local PufferLib 5.0 checkout (`./puffer sweep --help` and existing `config/*.ini` files) to confirm the exact section and key names for your version. Typical `.ini` structure:

```ini
[sweep]
method = protein
metric = sweep_score       # Logged in env Log struct (e.g., score or sweep_score)
goal = maximize
gpus = 4
total_experiments = 200

[protein]
batch_size = 4

[sweep.train.learning_rate]
distribution = log_normal
min = 3e-5
max = 3e-3
mean = 3e-4
scale = 0.5

[sweep.train.ent_coef]
distribution = log_normal
min = 1e-4
max = 5e-2
mean = 1e-2
scale = 0.5

[sweep.train.gamma]
distribution = logit_normal
min = 0.98
max = 0.9995
mean = 0.998
scale = 0.5

[sweep.env.num_agents]
distribution = int_uniform
min = 8
max = 14
mean = 11
scale = 1.0
```

Launch after `--debug --cpu`, baselines, and `--cu` have passed:

```bash
./build.sh <env> --cu
./puffer sweep <env> --sweep.method=protein --sweep.gpus=4
```

*(Override individual sweep parameters inline via dot notation, e.g., `--sweep.total_experiments=100`.)*

---

## 5. Reading Results in Constellation & Post-Sweep Validation

Never deploy a config straight off the top of a sweep table—Bayesian sweeps can pick lucky seeds or subtle exploits.

1. **Inspect the Pareto frontier in Constellation:** Plot `num_agents` vs. `delivered_per_hr`, `stalled_station_mins_per_hr`, and `min_soc` across all trials. Identify both the **minimum feasible `N`** and **`N + 1`** (your operational safety margin).
2. **Confirm reproducibility across seeds:** Re-run `./puffer train` on the top 2–3 winning configs across **3+ independent random seeds** (not just the lucky sweep seed).
3. **Run the mandatory `./puffer eval latest` multi-day audit:** Verify 3+ consecutive 24-hour cycles with zero SOC drift, zero collisions, zero off-limits intrusions, and throughput strictly within physical/demand bounds.
