# PufferLib 5.0 Industrial Automation RL Agent Bundle

Portable plugin/customization kit that encodes industrial throughput MDP design, PufferLib 5.0 C/CUDA memory and lifecycle invariants, automated static & baseline auditing, a `PreToolUse` GPU-training safety gate, and PuffeRL training/evaluation diagnostics.

## What's Inside

```text
pufferlib5-industrial-agent/
├── plugin.json                              # Plugin manifest
├── hooks.json                               # PreToolUse gate blocking --cu / puffer train until CPU debug + audit pass
├── rules/AGENTS.md                          # Always-on hard invariants
├── hooks/pre_train_gate.py                  # Hook script (Antigravity + Claude Code JSON contracts)
└── skills/pufferlib5-industrial-env/
    ├── SKILL.md                             # 6-phase industrial RL + PufferLib 5.0 runbook
    ├── references/
    │   ├── mdp_checklist.md                 # Markov state, reward, gamma & physics-ceiling guide
    │   ├── c_env_template.md                # Generic industrial C env template (env.h + binding.c)
    │   └── pufferl_diagnostics.md           # PuffeRL dashboard & ./puffer eval latest audit guide
    └── scripts/
        └── audit_env.py                     # CLI static & baseline verification tool
```

## Installation by Agent CLI

### Claude Code
```bash
# Project-local (run from your PufferLib 5.0 repo root; set KIT to your clone path)
KIT=/path/to/pufferlib5-industrial-agent
mkdir -p .claude/skills .claude/hooks
cp -R "$KIT/skills/pufferlib5-industrial-env" .claude/skills/
cp "$KIT/rules/AGENTS.md" ./CLAUDE.md
cp "$KIT/hooks/pre_train_gate.py" .claude/hooks/
```
Then add the `PreToolUse` hook to `.claude/settings.json`:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .claude/hooks/pre_train_gate.py"
          }
        ]
      }
    ]
  }
}
```
*(For global install in Claude Code, copy the skill into `~/.claude/skills/`.)*

### Codex
```bash
# Project-local (run from your PufferLib 5.0 repo root; set KIT to your clone path)
KIT=/path/to/pufferlib5-industrial-agent
mkdir -p .agents/skills
cp -R "$KIT/skills/pufferlib5-industrial-env" .agents/skills/
cp "$KIT/rules/AGENTS.md" ./AGENTS.md
```
*(For global install in Codex, copy the skill into `~/.agents/skills/`.)*

### Gemini CLI
```bash
# Project-local (run from your PufferLib 5.0 repo root; set KIT to your clone path)
KIT=/path/to/pufferlib5-industrial-agent
mkdir -p .gemini/skills
cp -R "$KIT/skills/pufferlib5-industrial-env" .gemini/skills/
cp "$KIT/rules/AGENTS.md" ./GEMINI.md
```
*(For global install in Gemini CLI, copy the skill into `~/.gemini/skills/`.)*

### Antigravity
```bash
# Global plugin install
KIT=/path/to/pufferlib5-industrial-agent
cp -R "$KIT" ~/.gemini/config/plugins/pufferlib5-industrial

# Or workspace-local plugin install (from your PufferLib 5.0 repo root)
mkdir -p .agents/plugins
cp -R "$KIT" .agents/plugins/pufferlib5-industrial
```

## Recommended Workflow
1. Describe your factory line (machines, buffers, cycle times, changeovers, wear/PM, routing) and invoke the `pufferlib5-industrial-env` skill.
2. Confirm the Markov spec and pen-and-paper physical throughput ceiling.
3. Let the agent write `ocean/<env>/<env>.h` and `binding.c`, run `audit_env.py static-check`, compile with `./build.sh <env> --debug --cpu`, and verify idle / random / heuristic baselines.
4. Once `audit_env.py check-baselines` stamps `.puffer_audit_passed.json`, the `PreToolUse` hook automatically unlocks `./build.sh <env> --cu` and `./puffer train <env>`.
5. Always finish with `./puffer eval latest` to verify the policy obeys physical bounds and hasn't exploited a simulator bug.
