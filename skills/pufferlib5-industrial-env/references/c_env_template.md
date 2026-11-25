# PufferLib 5.0 Native C Industrial Environment Template

Use this skeleton as a starting point for `ocean/<env>/<env>.h` and `ocean/<env>/binding.c`. Adapt the struct fields, transitions, and `binding.c` macro/registration calls to match the exact `vecenv.h` / `ocean/` template in your local PufferLib 5.0 checkout (inspect `ocean/squared/` or `ocean/target/` in your repo to confirm exact macro signatures).

## Shared Dimensions & Layout (`factory.h`)

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
    // Required PufferLib 5.0 flat shared-buffer pointers (sliced by runtime)
    Log log;
    float* observations;
    int*   actions;
    float* rewards;
    float* terminals;

    // Internal Markov state
    int mode;
    int timer;
    int current_job;     // 0=fast, 1=slow
    int setup_type;      // 0=fast, 1=slow (persisted across idle/PM)
    int tool_wear;       // 0..MAX_WEAR
    int buf_fast;        // waiting Fast jobs
    int buf_slow;        // waiting Slow jobs

    // Rolling KPI counters (NEVER exposed in observations)
    int window_ticks;
    int window_parts;
    int window_changeovers;
    int window_pm;
    int window_breakdowns;

    // Lock-free, CUDA-safe per-env PRNG state (seeded uniquely as base_seed + env_id)
    uint32_t rng;
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
    // 1. Explicitly clear all one-hot mode slots (PufferLib does NOT auto-zero buffers)
    for (int m = 0; m < NUM_MODES; m++) {
        env->observations[m] = 0.0f;
    }
    env->observations[env->mode] = 1.0f;

    // 2. Write normalized [0, 1] Markov state features (indices 5..9 -> OBS_DIM = 10)
    env->observations[5] = (float)env->timer / (float)BREAK_TICKS;
    env->observations[6] = (float)env->setup_type;
    env->observations[7] = (float)env->tool_wear / (float)MAX_WEAR;
    env->observations[8] = (float)env->buf_fast / (float)BUF_CAP;
    env->observations[9] = (float)env->buf_slow / (float)BUF_CAP;
}

// Internal reset helper — resets factory state & observations ONLY.
// Does NOT touch env->rewards[0] or env->terminals[0] so auto-reset inside c_step
// preserves the terminal flag and final step reward for PuffeRL.
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

// External reset entry point (called at env initialization with unique env->rng)
static inline void c_reset(FactoryEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;
    reset_internal_state(env);
}

static inline void c_step(FactoryEnv* env) {
    // 1. ALWAYS clear reward and terminal at the start of every tick
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    // 2. Apply policy action when idle (invalid actions are safe no-ops)
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

    // 3. Advance machine dynamics by 1 tick
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
            // Breakdown interrupts current job (explicitly documented rule)
            env->mode = MODE_BROKEN;
            env->timer = BREAK_TICKS;
            env->window_breakdowns++;
        } else if (env->timer <= 0) {
            // Finished good exits -> pure +1.0 throughput reward
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

    // 4. Stochastic arrivals into free buffer slots (lock-free per-env PRNG)
    if (env->buf_fast + env->buf_slow < BUF_CAP) {
        float u = rand_uniform(&env->rng);
        if (u < 0.35f) {
            env->buf_fast++;
        } else if (u < 0.60f) {
            env->buf_slow++;
        }
    }

    // 5. Rolling KPI logging (continuing task: terminals[0] stays 0.0f)
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

    // 6. If your environment DOES have a genuine fatal fault:
    // if (fatal_fault) {
    //     env->rewards[0] = -10.0f;
    //     env->terminals[0] = 1.0f;
    //     reset_internal_state(env); // Preserves rewards[0] and terminals[0]!
    //     return;
    // }

    // 7. Write updated observations
    write_observations(env);
}

#endif // FACTORY_H
```

## Binding Alignment Rules (`binding.c`)

When wiring `factory.h` into your PufferLib 5.0 `binding.c`:
1. `#include "factory.h"` and reference `FACTORY_OBS_DIM` and `FACTORY_ACT_DIM` directly (never hardcode magic numbers separately in `binding.c`).
2. Seed every parallel instance uniquely during initialization:
   ```c
   // Inside init loop for factory i:
   env->rng = (uint32_t)(seed + i * 2654435761u + 1u);
   ```
3. Verify your local PufferLib 5.0 repo's `ocean/<reference_env>/binding.c` (e.g., `squared` or `target`) for the exact `vecenv.h` / Python-binding macro names (`my_init`, `my_log`, `Env` typedef) before compiling.
