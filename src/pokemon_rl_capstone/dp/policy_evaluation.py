"""Iterative policy evaluation for tabular MDP."""

from typing import Callable, Dict, List, Tuple

Transition = Tuple[float, int, float, bool]
PFunc = Callable[[int, int], List[Transition]]

def policy_evaluation(
    states: list[int],
    actions: list[int],
    P: PFunc,
    policy: Callable[[int], Dict[int, float]],
    gamma: float = 0.98,
    tol: float = 1e-8,
    max_iters: int = 10_000,
):
    raise NotImplementedError
