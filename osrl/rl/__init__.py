"""osrl.rl — RL policy package (MaskablePPO + Gymnasium env + inference bridge).

Structural guarantees:
- This package is NEVER imported by osrl.guard, osrl.worker, osrl.executor,
  osrl.pidfd, or osrl.daemon (enforced by test_safety_invariants.py AST scan).
- The guard chain remains a total function of machine state: no RL inference
  on the safety path, ever.
- The inference bridge writes to the store's intents table; the guard/worker/
  executor chain applies actions independently and can always veto.
"""
