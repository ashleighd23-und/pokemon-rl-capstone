"""Iterative policy evaluation for tabular MDP."""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple

Transition = Tuple[float, int, float, bool]
PFunc = Callable[[int, int], List[Transition]]
PolicyFunc = Callable[[int], Dict[int, float]]

def policy_evaluation(
    states: list[int],
    actions: list[int],
    P: PFunc,
    policy: PolicyFunc,
    gamma: float = 0.98,
    tol: float = 1e-10,
    max_iters: int = 50_000,
):
    V = {s: 0.0 for s in states}

    for _ in range(max_iters):
        delta = 0.0
        for s in states:
            pi = policy(s)
            if not pi:
                continue

            v_new = 0.0
            for a, pa in pi.items():
                if pa <= 0:
                    continue
                for p, s2, r, done in P(s, a):
                    v_new += pa * p * (r + (0.0 if done else gamma * V[s2]))

            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new

        if delta < tol:
            break

    return V
