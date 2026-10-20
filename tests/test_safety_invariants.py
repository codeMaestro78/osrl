"""Structural safety invariants — enforced mechanically, not by convention."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The safety path must be free of RL/model imports, forever, by construction.
FORBIDDEN_MODULE_ROOTS = (
    "torch", "stable_baselines", "sb3_contrib", "gymnasium", "gym",
    "sklearn", "keras", "tensorflow", "transformers",
)
GUARD_CHAIN_MODULES = (
    "guard.py", "worker.py", "executor.py", "pidfd.py", "procid.py",
)


def imported_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_guard_chain_has_no_rl_imports():
    for name in GUARD_CHAIN_MODULES:
        path = REPO / "osrl" / name
        roots = imported_roots(path)
        bad = sorted(roots & set(FORBIDDEN_MODULE_ROOTS))
        assert not bad, f"{name} imports RL/model modules: {bad}"


def test_daemon_never_imports_worker():
    """The collector daemon must never start autonomous execution."""
    tree = ast.parse((REPO / "osrl" / "daemon.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("worker" in m for m in imported), imported


def test_executor_is_the_only_mutation_surface():
    """osrl.executor must be the only module referencing the OS mutation
    API; other modules may READ state but never mutate."""
    offenders = []
    for path in (REPO / "osrl").glob("*.py"):
        if path.name == "executor.py":
            continue
        if "setpriority" in path.read_text():
            offenders.append(path.name)
    assert offenders == [], f"mutation API leaked into: {offenders}"


def test_adapters_never_write_nice():
    src = (REPO / "osrl" / "adapters.py").read_text()
    assert "setpriority" not in src
    # a mutating call would read ".nice(" with an argument; adapters only
    # READ cpu times (nice field name) and never call Process.nice(arg)
    assert "nice(" not in src.replace("nice=0.0", "")
    assert "renice" not in src
