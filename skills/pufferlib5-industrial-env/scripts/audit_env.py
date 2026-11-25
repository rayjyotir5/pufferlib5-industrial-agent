#!/usr/bin/env python3
"""PufferLib 5.0 Industrial Automation Environment Auditor.

Stdlib-only CLI for static C/binding checks, CPU debug build stamping,
and non-RL baseline physics-ceiling verification.

Usage:
  uv run scripts/audit_env.py static-check --env-dir ocean/<env> --output /tmp/static.json
  uv run scripts/audit_env.py record-cpu-pass --env-dir ocean/<env> --output /tmp/cpu.json
  uv run scripts/audit_env.py check-baselines --env-dir ocean/<env> ... --output /tmp/baselines.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAMP_FILENAME = ".puffer_audit_passed.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_env_files(env_dir: Path) -> tuple[list[Path], Path | None]:
    headers = sorted(env_dir.glob("*.h"))
    binding = env_dir / "binding.c"
    return headers, binding if binding.exists() else None


def compute_source_hashes(env_dir: Path) -> dict[str, str]:
    headers, binding = find_env_files(env_dir)
    hashes: dict[str, str] = {}
    for h in headers:
        hashes[h.name] = sha256_file(h)
    if binding is not None:
        hashes[binding.name] = sha256_file(binding)
    return hashes


def strip_c_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    src = re.sub(r"//.*", "", src)
    return src


def extract_function_body(src: str, func_name: str) -> str | None:
    pattern = re.compile(rf"\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{", re.MULTILINE)
    m = pattern.search(src)
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    for idx in range(start, len(src)):
        ch = src[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start : idx + 1]
    return src[start:]


def load_stamp(env_dir: Path) -> dict[str, Any]:
    stamp_path = env_dir / STAMP_FILENAME
    if stamp_path.exists():
        try:
            return json.loads(stamp_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_stamp(env_dir: Path, data: dict[str, Any]) -> Path:
    stamp_path = env_dir / STAMP_FILENAME
    stamp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return stamp_path


def write_report(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cmd_static_check(args: argparse.Namespace) -> int:
    env_dir = Path(args.env_dir).resolve()
    output_path = Path(args.output).resolve()

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not env_dir.is_dir():
        report = {
            "status": "fail",
            "env_dir": str(env_dir),
            "errors": [{"code": "MISSING_DIR", "message": f"Directory not found: {env_dir}"}],
            "warnings": [],
        }
        write_report(output_path, report)
        print(f"FAIL: Directory not found ({env_dir}). Report written to: {output_path}", file=sys.stderr)
        return 1

    headers, binding = find_env_files(env_dir)
    if not headers:
        errors.append({"code": "MISSING_HEADER", "message": "No .h environment header found in env_dir."})
    if binding is None:
        errors.append({"code": "MISSING_BINDING", "message": "No binding.c found in env_dir."})

    header_obs_macros: dict[str, int] = {}
    header_act_macros: dict[str, int] = {}
    found_c_step = False
    found_c_reset = False

    for hdr in headers:
        raw = hdr.read_text(encoding="utf-8", errors="replace")
        clean = strip_c_comments(raw)

        # Capture OBS_DIM / ACT_DIM style #defines
        for m in re.finditer(r"#\s*define\s+([A-Z0-9_]*OBS_DIM[A-Z0-9_]*)\s+(\d+)", clean):
            header_obs_macros[m.group(1)] = int(m.group(2))
        for m in re.finditer(r"#\s*define\s+([A-Z0-9_]*ACT_DIM[A-Z0-9_]*)\s+(\d+)", clean):
            header_act_macros[m.group(1)] = int(m.group(2))

        # Check global rand()/srand()/random() anywhere in header
        if re.search(r"\b(?:rand|srand|random|srandom)\s*\(", clean):
            errors.append({
                "code": "GLOBAL_RAND",
                "file": hdr.name,
                "message": "Found libc rand()/srand()/random(). Use a lock-free per-env PRNG (e.g., xorshift32(&env->rng)) seeded uniquely per env.",
            })

        # Check c_step invariants
        step_body = extract_function_body(clean, "c_step")
        if step_body is not None:
            found_c_step = True
            if re.search(r"\b(?:malloc|calloc|realloc|free)\s*\(", step_body):
                errors.append({
                    "code": "ALLOC_IN_STEP",
                    "file": hdr.name,
                    "message": "Dynamic memory allocation (malloc/calloc/realloc/free) is forbidden inside c_step.",
                })
            if not re.search(r"rewards\s*(?:\[\s*0\s*\]|->|\[[^\]]+\])\s*=\s*0(?:\.0f?)?\s*;", step_body):
                errors.append({
                    "code": "REWARD_NOT_CLEARED",
                    "file": hdr.name,
                    "message": "c_step must explicitly zero env->rewards[...] = 0.0f at the start of every step to prevent sticky rewards.",
                })
            if not re.search(r"terminals\s*(?:\[\s*0\s*\]|->|\[[^\]]+\])\s*=\s*0(?:\.0f?)?\s*;", step_body):
                errors.append({
                    "code": "TERMINAL_NOT_CLEARED",
                    "file": hdr.name,
                    "message": "c_step must explicitly zero env->terminals[...] = 0.0f at the start of every step.",
                })

            # Check if c_step calls c_reset directly while c_reset zeroes rewards/terminals
            calls_c_reset = bool(re.search(r"\bc_reset\s*\(", step_body))
            reset_body = extract_function_body(clean, "c_reset")
            if calls_c_reset and reset_body is not None:
                if re.search(r"(?:rewards|terminals)\s*\[[^\]]+\]\s*=\s*0", reset_body):
                    errors.append({
                        "code": "RESET_WIPES_TERMINAL_REWARD",
                        "file": hdr.name,
                        "message": "c_step calls c_reset(), which zeroes rewards/terminals. Extract reset_internal_state() so auto-reset preserves terminal=1 and final reward.",
                    })

        reset_body = extract_function_body(clean, "c_reset")
        if reset_body is not None:
            found_c_reset = True
            if re.search(r"->\s*rng\s*=\s*\d+\s*;", reset_body):
                errors.append({
                    "code": "CONSTANT_RNG_SEED",
                    "file": hdr.name,
                    "message": "c_reset assigns a constant literal to env->rng, making all parallel environments identical. Seed uniquely per env_id during initialization.",
                })

        # Warn if observations are written without any zeroing/memset in header
        if re.search(r"observations\s*\[", clean):
            has_zeroing = bool(
                re.search(r"memset\s*\([^;]*observations", clean)
                or re.search(r"observations\s*\[[^\]]+\]\s*=\s*0(?:\.0f?)?\s*;", clean)
            )
            if not has_zeroing:
                warnings.append({
                    "code": "OBS_ZEROING_CHECK",
                    "file": hdr.name,
                    "message": "No explicit zeroing (memset or = 0.0f) of env->observations detected. Ensure all one-hot/sparse slots are cleared every tick.",
                })

    if headers and not found_c_step:
        errors.append({"code": "MISSING_C_STEP", "message": "Could not find c_step(...) definition in environment header(s)."})
    if headers and not found_c_reset:
        warnings.append({"code": "MISSING_C_RESET", "message": "Could not find c_reset(...) definition in environment header(s)."})

    # Cross-check binding.c against header OBS_DIM / ACT_DIM if present
    if binding is not None:
        b_clean = strip_c_comments(binding.read_text(encoding="utf-8", errors="replace"))
        if re.search(r"\b(?:rand|srand)\s*\(", b_clean):
            warnings.append({
                "code": "RAND_IN_BINDING",
                "file": binding.name,
                "message": "Found rand()/srand() in binding.c; prefer deterministic per-env seeding (seed + env_id).",
            })
        if re.search(r"->\s*rng\s*=\s*\d+\s*;", b_clean):
            errors.append({
                "code": "CONSTANT_RNG_SEED_BINDING",
                "file": binding.name,
                "message": "binding.c assigns a constant literal to env->rng. Seed each parallel env uniquely (e.g., seed + i).",
            })
        for macro_name, val in header_obs_macros.items():
            # Check if binding.c redefines the same macro with a different value
            m = re.search(rf"#\s*define\s+{re.escape(macro_name)}\s+(\d+)", b_clean)
            if m and int(m.group(1)) != val:
                errors.append({
                    "code": "OBS_DIM_MISMATCH",
                    "file": binding.name,
                    "message": f"{macro_name} is {val} in header but {m.group(1)} in binding.c.",
                })
        for macro_name, val in header_act_macros.items():
            m = re.search(rf"#\s*define\s+{re.escape(macro_name)}\s+(\d+)", b_clean)
            if m and int(m.group(1)) != val:
                errors.append({
                    "code": "ACT_DIM_MISMATCH",
                    "file": binding.name,
                    "message": f"{macro_name} is {val} in header but {m.group(1)} in binding.c.",
                })

    passed = len(errors) == 0
    source_hashes = compute_source_hashes(env_dir)

    if passed:
        stamp = load_stamp(env_dir)
        if stamp.get("source_hashes") != source_hashes:
            stamp = {"source_hashes": source_hashes}
        stamp["static_check_passed"] = True
        stamp["static_check_at"] = datetime.now(timezone.utc).isoformat()
        save_stamp(env_dir, stamp)

    report = {
        "status": "pass" if passed else "fail",
        "env_dir": str(env_dir),
        "source_hashes": source_hashes,
        "header_obs_macros": header_obs_macros,
        "header_act_macros": header_act_macros,
        "errors": errors,
        "warnings": warnings,
    }
    write_report(output_path, report)

    if passed:
        print(f"PASS: static-check succeeded ({len(warnings)} warnings). Report written to: {output_path}")
        return 0
    print(f"FAIL: static-check found {len(errors)} error(s). Report written to: {output_path}", file=sys.stderr)
    return 1


def cmd_record_cpu_pass(args: argparse.Namespace) -> int:
    env_dir = Path(args.env_dir).resolve()
    output_path = Path(args.output).resolve()

    if not env_dir.is_dir():
        write_report(output_path, {"status": "fail", "error": f"Directory not found: {env_dir}"})
        print(f"FAIL: Directory not found: {env_dir}", file=sys.stderr)
        return 1

    source_hashes = compute_source_hashes(env_dir)
    if not source_hashes:
        write_report(output_path, {"status": "fail", "error": "No .h or binding.c files found to hash."})
        print("FAIL: No .h or binding.c files found.", file=sys.stderr)
        return 1

    stamp = load_stamp(env_dir)
    if stamp.get("source_hashes") != source_hashes:
        # Preserve only if hashes match; otherwise reset downstream gates
        stamp = {"source_hashes": source_hashes, "static_check_passed": False, "baselines_passed": False}

    stamp["cpu_debug_passed"] = True
    stamp["cpu_debug_at"] = datetime.now(timezone.utc).isoformat()
    stamp_path = save_stamp(env_dir, stamp)

    report = {
        "status": "pass",
        "env_dir": str(env_dir),
        "stamp_path": str(stamp_path),
        "source_hashes": source_hashes,
        "cpu_debug_passed": True,
    }
    write_report(output_path, report)
    print(f"PASS: Recorded --debug --cpu pass for {env_dir.name}. Report written to: {output_path}")
    return 0


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def cmd_check_baselines(args: argparse.Namespace) -> int:
    env_dir = Path(args.env_dir).resolve()
    output_path = Path(args.output).resolve()

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if args.fastest_cycle_ticks <= 0:
        errors.append({"code": "INVALID_CYCLE", "message": "fastest_cycle_ticks must be > 0."})
    if args.horizon <= 0:
        errors.append({"code": "INVALID_HORIZON", "message": "horizon must be > 0."})
    if args.exit_machines <= 0:
        errors.append({"code": "INVALID_MACHINES", "message": "exit_machines must be > 0."})

    ceiling = (
        math.floor(args.horizon / args.fastest_cycle_ticks) * args.exit_machines
        if args.fastest_cycle_ticks > 0 and args.horizon > 0 and args.exit_machines > 0
        else 0
    )

    if abs(args.idle_throughput) > 1e-6:
        errors.append({
            "code": "NONZERO_IDLE",
            "message": f"Always-idle throughput is {args.idle_throughput}, expected 0. Check for phantom completions or sticky rewards.",
        })
    if args.has_nan:
        errors.append({
            "code": "NAN_DETECTED",
            "message": "NaN/Inf detected during random rollouts. Check division-by-zero and buffer bounds under --debug --cpu.",
        })
    if args.obs_out_of_bounds:
        errors.append({
            "code": "OBS_OUT_OF_BOUNDS",
            "message": "Observations exceeded expected normalized bounds. Check normalization and one-hot clearing.",
        })
    if args.random_throughput > ceiling + 1e-6:
        errors.append({
            "code": "RANDOM_EXCEEDS_CEILING",
            "message": f"Random throughput ({args.random_throughput}) exceeds physical ceiling ({ceiling}). Check for double-counting, off-by-one timers, or sticky rewards.",
        })
    if args.heuristic_throughput > ceiling + 1e-6:
        errors.append({
            "code": "HEURISTIC_EXCEEDS_CEILING",
            "message": f"Heuristic throughput ({args.heuristic_throughput}) exceeds physical ceiling ({ceiling}). Physics or reward bug confirmed.",
        })
    if args.heuristic_throughput < args.random_throughput:
        warnings.append({
            "code": "HEURISTIC_BELOW_RANDOM",
            "message": f"Domain heuristic ({args.heuristic_throughput}) scored below random ({args.random_throughput}); inspect heuristic logic or action mapping.",
        })

    passed = len(errors) == 0
    source_hashes = compute_source_hashes(env_dir) if env_dir.is_dir() else {}

    if passed and env_dir.is_dir():
        stamp = load_stamp(env_dir)
        if stamp.get("source_hashes") != source_hashes:
            stamp = {"source_hashes": source_hashes, "static_check_passed": False, "cpu_debug_passed": False}
        stamp["baselines_passed"] = True
        stamp["baselines_at"] = datetime.now(timezone.utc).isoformat()
        stamp["physical_ceiling"] = ceiling
        stamp["benchmark_heuristic_throughput"] = args.heuristic_throughput
        save_stamp(env_dir, stamp)

    report = {
        "status": "pass" if passed else "fail",
        "env_dir": str(env_dir),
        "physical_ceiling": ceiling,
        "horizon": args.horizon,
        "fastest_cycle_ticks": args.fastest_cycle_ticks,
        "exit_machines": args.exit_machines,
        "idle_throughput": args.idle_throughput,
        "random_throughput": args.random_throughput,
        "heuristic_throughput": args.heuristic_throughput,
        "errors": errors,
        "warnings": warnings,
    }
    write_report(output_path, report)

    if passed:
        print(
            f"PASS: Baselines valid (ceiling={ceiling}, random={args.random_throughput}, "
            f"heuristic={args.heuristic_throughput}). Report written to: {output_path}"
        )
        return 0
    print(f"FAIL: Baseline check failed with {len(errors)} error(s). Report written to: {output_path}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit PufferLib 5.0 industrial C environments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_static = sub.add_parser("static-check", help="Run static C/binding safety checks.")
    p_static.add_argument("--env-dir", required=True, help="Path to ocean/<env> directory.")
    p_static.add_argument("--output", required=True, help="Path to write JSON report.")
    p_static.set_defaults(func=cmd_static_check)

    p_cpu = sub.add_parser("record-cpu-pass", help="Record that ./build.sh <env> --debug --cpu passed.")
    p_cpu.add_argument("--env-dir", required=True, help="Path to ocean/<env> directory.")
    p_cpu.add_argument("--output", required=True, help="Path to write JSON report.")
    p_cpu.set_defaults(func=cmd_record_cpu_pass)

    p_base = sub.add_parser("check-baselines", help="Validate idle, random, and heuristic baselines against physical ceiling.")
    p_base.add_argument("--env-dir", required=True, help="Path to ocean/<env> directory.")
    p_base.add_argument("--fastest-cycle-ticks", type=int, required=True, help="Shortest possible job cycle time (ticks).")
    p_base.add_argument("--horizon", type=int, required=True, help="Evaluation horizon (ticks), e.g., 1000.")
    p_base.add_argument("--exit-machines", type=int, default=1, help="Number of parallel exit machines (default: 1).")
    p_base.add_argument("--idle-throughput", type=float, required=True, help="Throughput under always-idle policy.")
    p_base.add_argument("--random-throughput", type=float, required=True, help="Throughput under uniform random policy.")
    p_base.add_argument("--heuristic-throughput", type=float, required=True, help="Throughput under simple domain heuristic.")
    p_base.add_argument("--has-nan", type=parse_bool, required=True, help="Whether any NaN/Inf appeared (true/false).")
    p_base.add_argument("--obs-out-of-bounds", type=parse_bool, required=True, help="Whether any obs left expected bounds (true/false).")
    p_base.add_argument("--output", required=True, help="Path to write JSON report.")
    p_base.set_defaults(func=cmd_check_baselines)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
