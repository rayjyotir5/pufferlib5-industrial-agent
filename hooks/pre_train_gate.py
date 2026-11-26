#!/usr/bin/env python3
"""PreToolUse safety gate for PufferLib 5.0 CUDA builds and training.

Blocks `./build.sh <env> --cu` and `./puffer train <env>` unless
`ocean/<env>/.puffer_audit_passed.json` exists, confirms static-check +
`--debug --cpu` + baselines all passed, AND matches current hashes of
`<env>.h` and `binding.c`.

Supports both Antigravity hooks.json payloads and Claude Code PreToolUse payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

STAMP_FILENAME = ".puffer_audit_passed.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def current_hashes(env_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for hdr in sorted(env_dir.glob("*.h")):
        hashes[hdr.name] = sha256_file(hdr)
    binding = env_dir / "binding.c"
    if binding.exists():
        hashes[binding.name] = sha256_file(binding)
    return hashes


def extract_command_and_cwd(payload: dict[str, Any]) -> tuple[str, Path, bool]:
    """Return (command_line, cwd, is_claude_code_format)."""
    # Antigravity format: toolCall.name == "run_command", toolCall.args.CommandLine
    if "toolCall" in payload:
        tc = payload.get("toolCall") or {}
        args = tc.get("args") or {}
        cmd = str(args.get("CommandLine") or args.get("command") or "")
        cwd = Path(payload.get("cwd") or ".")
        return cmd, cwd, False

    # Claude Code format: tool_name == "Bash", tool_input.command
    tool_input = payload.get("tool_input") or {}
    cmd = str(tool_input.get("command") or "")
    cwd = Path(payload.get("cwd") or ".")
    return cmd, cwd, True


def emit_decision(allow: bool, reason: str, is_claude: bool) -> int:
    if is_claude:
        if allow:
            print(json.dumps({"continue": True}))
            return 0
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }))
        return 0

    # Antigravity hooks.json contract
    out: dict[str, Any] = {"decision": "allow" if allow else "deny"}
    if reason:
        out["reason"] = reason
    print(json.dumps(out))
    return 0


def detect_gated_env(cmd: str) -> tuple[str | None, bool]:
    """Detect `./build.sh <env> ... --cu`, `./puffer train <env>`, or `./puffer sweep <env>`.

    Returns (env_name_or_none, is_gated_command).
    """
    m_build = re.search(r"(?:^|[;&|]\s*)\.?/?build\.sh\s+([A-Za-z0-9_-]+)\b([^\n;&|]*)", cmd)
    if m_build and re.search(r"(?:^|\s)--cu(?:\s|$)", m_build.group(2)):
        return m_build.group(1), True

    m_train = re.search(r"(?:^|[;&|]\s*)\.?/?puffer\s+(?:train|sweep)(?:\s+([A-Za-z0-9_-]+))?", cmd)
    if m_train:
        env_name = m_train.group(1)
        if env_name and env_name.startswith("-"):
            env_name = None
        return env_name, True

    return None, False


def find_env_dir(cwd: Path, env_name: str | None) -> Path | None:
    if env_name:
        for candidate in [cwd / "ocean" / env_name, cwd / "pufferlib" / "ocean" / env_name, cwd / env_name]:
            if candidate.is_dir():
                return candidate
        return None

    # If `./puffer train` had no explicit env name, check if a single stamped env exists under ocean/
    ocean = cwd / "ocean"
    if ocean.is_dir():
        stamps = list(ocean.glob(f"*/{STAMP_FILENAME}"))
        if len(stamps) == 1:
            return stamps[0].parent
    return None


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        return emit_decision(True, "", is_claude=False)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return emit_decision(True, "", is_claude=False)

    cmd, cwd, is_claude = extract_command_and_cwd(payload)
    env_name, is_gated = detect_gated_env(cmd)
    if not is_gated:
        return emit_decision(True, "", is_claude=is_claude)

    env_dir = find_env_dir(cwd, env_name)
    if env_dir is None:
        # Don't block built-in envs outside custom ocean/<env> directories when env_dir cannot be resolved
        if env_name is None:
            return emit_decision(True, "", is_claude=is_claude)
        return emit_decision(
            False,
            f"Blocked GPU build/training for '{env_name}': could not locate ocean/{env_name}/ to verify pre-training audit stamp. "
            f"Run static-check, `./build.sh {env_name} --debug --cpu`, record-cpu-pass, and check-baselines first.",
            is_claude=is_claude,
        )

    stamp_path = env_dir / STAMP_FILENAME
    if not stamp_path.exists():
        return emit_decision(
            False,
            f"Blocked GPU build/training for '{env_dir.name}': missing {STAMP_FILENAME}. "
            f"Run audit_env.py static-check, `./build.sh {env_dir.name} --debug --cpu`, "
            f"audit_env.py record-cpu-pass, and audit_env.py check-baselines first.",
            is_claude=is_claude,
        )

    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception:
        return emit_decision(
            False,
            f"Blocked GPU build/training for '{env_dir.name}': corrupted {STAMP_FILENAME}. Re-run audit_env.py checks.",
            is_claude=is_claude,
        )

    missing = [
        gate
        for gate in ("static_check_passed", "cpu_debug_passed", "baselines_passed")
        if not stamp.get(gate)
    ]
    if missing:
        return emit_decision(
            False,
            f"Blocked GPU build/training for '{env_dir.name}': incomplete pre-training gates: {', '.join(missing)}.",
            is_claude=is_claude,
        )

    live_hashes = current_hashes(env_dir)
    if stamp.get("source_hashes") != live_hashes:
        return emit_decision(
            False,
            f"Blocked GPU build/training for '{env_dir.name}': source files changed since the last CPU debug + audit pass. "
            f"Re-run static-check, `./build.sh {env_dir.name} --debug --cpu`, record-cpu-pass, and check-baselines.",
            is_claude=is_claude,
        )

    return emit_decision(True, "", is_claude=is_claude)


if __name__ == "__main__":
    raise SystemExit(main())
