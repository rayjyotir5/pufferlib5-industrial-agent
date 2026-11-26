# PufferLib 5.0 Native C Industrial Environment Templates

Choose **Pattern A** for short-horizon single-machine/cell environments, or **Pattern B** for multi-agent fleets / multi-hour shift-and-charging problems requiring macro-stepping over a deterministic 1-tick C physics kernel. Always inspect `ocean/squared/` or `ocean/target/` in your local PufferLib 5.0 checkout for exact `binding.c` / `vecenv.h` macro signatures.

---

## Pattern A: Single-Machine 1-Tick Cell (`factory.h`)

```c
#ifndef FACTORY_H
#define FACTORY_H

#include <stdint.h>
#include <string.h>

// Single source of truth for dimensions — binding.c MUST use these exact constants.
#define FACTORY_OBS_DIM 10
#define FACTORY_ACT_DIM 4   // 0=idle, 1=load_fast, 2=load_slow, 3=start_pm

#define MODE_IDLE      0
#define MODE_PROC      1
#define MODE_SETUP     2
#define MODE_PM        3
#define MODE_BROKEN    4
#define NUM_MODES      5

#define BUF_CAP        3
#define MAX_WEAR       20
#define FAST_TICKS     2
#define SLOW_TICKS     5
#define SETUP_TICKS    2
#define PM_TICKS       3
#define BREAK_TICKS    10
#define LOG_WINDOW     1000

typedef struct {
    float parts_per_window;
    float changeovers;
    float pm_count;
    float breakdowns;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    int*   actions;
    float* rewards;
    float* terminals;

    int mode;
    int timer;
    int current_job;     // 0=fast, 1=slow
    int setup_type;      // 0=fast, 1=slow (persisted across idle/PM)
    int tool_wear;       // 0..MAX_WEAR
    int buf_fast;
    int buf_slow;

    int window_ticks;
    int window_parts;
    int window_changeovers;
    int window_pm;
    int window_breakdowns;

    uint32_t rng;        // Per-env PRNG state (seeded uniquely: base_seed + env_id)
} FactoryEnv;

static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    if (x == 0) x = 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand_uniform(uint32_t* state) {
    return (xorshift32(state) & 0x00FFFFFFu) / 16777216.0f;
}

static inline void write_observations(FactoryEnv* env) {
    for (int m = 0; m < NUM_MODES; m++) {
        env->observations[m] = 0.0f;
    }
    env->observations[env->mode] = 1.0f;
    env->observations[5] = (float)env->timer / (float)BREAK_TICKS;
    env->observations[6] = (float)env->setup_type;
    env->observations[7] = (float)env->tool_wear / (float)MAX_WEAR;
    env->observations[8] = (float)env->buf_fast / (float)BUF_CAP;
    env->observations[9] = (float)env->buf_slow / (float)BUF_CAP;
}

// Resets internal state & initial observations ONLY — preserves rewards/terminals on auto-reset.
static inline void reset_internal_state(FactoryEnv* env) {
    env->mode = MODE_IDLE;
    env->timer = 0;
    env->current_job = 0;
    env->setup_type = 0;
    env->tool_wear = 0;
    env->buf_fast = 1;
    env->buf_slow = 1;
    env->window_ticks = 0;
    env->window_parts = 0;
    env->window_changeovers = 0;
    env->window_pm = 0;
    env->window_breakdowns = 0;
    write_observations(env);
}

static inline void c_reset(FactoryEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    reset_internal_state(env);
}

static inline void c_step(FactoryEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    int act = env->actions[0];
    if (env->mode == MODE_IDLE) {
        if (act == 1 && env->buf_fast > 0) {
            env->buf_fast--;
            env->current_job = 0;
            if (env->setup_type != 0) {
                env->mode = MODE_SETUP;
                env->timer = SETUP_TICKS;
                env->window_changeovers++;
            } else {
                env->mode = MODE_PROC;
                env->timer = FAST_TICKS;
            }
        } else if (act == 2 && env->buf_slow > 0) {
            env->buf_slow--;
            env->current_job = 1;
            if (env->setup_type != 1) {
                env->mode = MODE_SETUP;
                env->timer = SETUP_TICKS;
                env->window_changeovers++;
            } else {
                env->mode = MODE_PROC;
                env->timer = SLOW_TICKS;
            }
        } else if (act == 3) {
            env->mode = MODE_PM;
            env->timer = PM_TICKS;
            env->window_pm++;
        }
    }

    if (env->mode == MODE_SETUP) {
        env->timer--;
        if (env->timer <= 0) {
            env->setup_type = env->current_job;
            env->mode = MODE_PROC;
            env->timer = (env->current_job == 0) ? FAST_TICKS : SLOW_TICKS;
        }
    } else if (env->mode == MODE_PROC) {
        env->timer--;
        env->tool_wear++;
        if (env->tool_wear >= MAX_WEAR) {
            env->mode = MODE_BROKEN;
            env->timer = BREAK_TICKS;
            env->window_breakdowns++;
        } else if (env->timer <= 0) {
            env->rewards[0] = 1.0f;
            env->window_parts++;
            env->mode = MODE_IDLE;
            env->timer = 0;
        }
    } else if (env->mode == MODE_PM || env->mode == MODE_BROKEN) {
        env->timer--;
        if (env->timer <= 0) {
            env->tool_wear = 0;
            env->mode = MODE_IDLE;
            env->timer = 0;
        }
    }

    if (env->buf_fast + env->buf_slow < BUF_CAP) {
        float u = rand_uniform(&env->rng);
        if (u < 0.35f) env->buf_fast++;
        else if (u < 0.60f) env->buf_slow++;
    }

    env->window_ticks++;
    if (env->window_ticks >= LOG_WINDOW) {
        env->log.parts_per_window += (float)env->window_parts;
        env->log.changeovers += (float)env->window_changeovers;
        env->log.pm_count += (float)env->window_pm;
        env->log.breakdowns += (float)env->window_breakdowns;
        env->log.n += 1.0f;
        env->window_ticks = 0;
        env->window_parts = 0;
        env->window_changeovers = 0;
        env->window_pm = 0;
        env->window_breakdowns = 0;
    }

    write_observations(env);
}

#endif // FACTORY_H
```

---

## Pattern B: Multi-Agent Lockstep Macro-Step Fleet (`num_agents = N`, `K`-Tick Inner C Physics)

Use this structural pattern when coordinating `N` mobile agents (e.g., AMRs/AGVs/cranes) across multi-hour shifts and charging cycles:

```c
#define MAX_AGENTS     16
#define MACRO_K        30        // 30 physical ticks per RL step
#define FLEET_OBS_DIM  62
#define FLEET_ACT_DIM  13
#define TERMINAL_PEN   -100.0f   // Must satisfy |TERMINAL_PEN| > max_step_penalty / (1 - gamma)

// Inside FleetEnv struct (all arrays fixed-size — ZERO malloc in c_step):
//   int num_agents;
//   float* observations; // contiguous slab of length num_agents * FLEET_OBS_DIM
//   int*   actions;      // contiguous slab of length num_agents
//   float* rewards;      // contiguous slab of length num_agents
//   float* terminals;    // contiguous slab of length num_agents
//   int    occ_grid[GRID_W][GRID_H]; // -1 if empty, else agent index a
//   AgentState agents[MAX_AGENTS];   // (x, y, heading, mode, dwell, battery, next_action)
//   uint32_t rng;

static inline void fleet_c_step(FleetEnv* env) {
    // 1. Clear ALL agents' rewards & terminals at the start of every macro-step
    for (int a = 0; a < env->num_agents; a++) {
        env->rewards[a] = 0.0f;
        env->terminals[a] = 0.0f;
    }

    // 2. Apply policy actions:
    //    - Idle / charging / parked agents can switch actions immediately (interruptible charging!)
    //    - Busy agents on a pick/drop mission store actions[a] as next_action (zero idle gap!)
    apply_macro_actions(env);

    // 3. Advance exact 1-tick physics for MACRO_K ticks
    float team_reward = 0.0f;
    int hard_failure = 0;

    for (int sub = 0; sub < MACRO_K; sub++) {
        // 3a. Stochastic arrivals during active shift; pause & flag stations at BUF_CAP
        int stalled_stations = step_arrivals_and_stalls(env);
        if (is_active_shift(env)) {
            team_reward -= (0.10f / (float)MACRO_K) * (float)stalled_stations;
        }

        // 3b. Deterministic 1-tick traffic & conflict resolution (PIBT: vertex + no-edge-swap)
        //     Priority: low-battery-to-charger > loaded > heading-to-pickup > parked (auto-yield)
        //     Movement ticks drain battery -= 1; turns/dwells drain 0; chargers add +2.
        int delivered = 0;
        if (step_physics_pibt_one_tick(env, &delivered) != 0) {
            hard_failure = 1; // e.g., an AMR hit 0% battery mid-corridor
            break;
        }
        team_reward += (float)delivered;

        // 3c. Immediately dispatch pre-assigned next_action for any AMR that just finished a drop
        dispatch_finished_agents_from_preassigned(env);
        env->tick++;
    }

    // 4. Write cooperative team reward (or hard terminal penalty) across all N agents
    if (hard_failure) {
        for (int a = 0; a < env->num_agents; a++) {
            env->rewards[a] = TERMINAL_PEN;
            env->terminals[a] = 1.0f;
        }
        reset_internal_fleet_state(env); // Preserves rewards & terminals!
        return;
    }

    float per_agent_rew = team_reward / (float)env->num_agents;
    for (int a = 0; a < env->num_agents; a++) {
        env->rewards[a] = per_agent_rew;
        // Write normalized [0, 1] obs slice at env->observations + a * FLEET_OBS_DIM
        // explicitly zeroing all one-hot slots first!
        write_agent_observations(env, a);
    }
}
```

---

## Binding Alignment Rules (`binding.c`)

1. `#include "<env>.h"` and reference `*_OBS_DIM` and `*_ACT_DIM` directly (never duplicate magic integers in `binding.c`).
2. Seed every parallel env uniquely during initialization:
   ```c
   env->rng = (uint32_t)(seed + i * 2654435761u + 1u);
   ```
3. For multi-agent envs (`num_agents = N`), confirm that `binding.c` allocates/slices `N` agent rows per environment (`env_i` owns `N * OBS_DIM` observation floats and `N` actions/rewards/terminals).
