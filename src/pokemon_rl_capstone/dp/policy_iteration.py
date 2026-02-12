"""Policy iteration (policy evaluation + greedy improvement)."""

from typing import Callable, Dict, List, Tuple
from .policy_evaluation import policy_evaluation, PFunc

def policy_iteration(
    states: list[int],
    actions: list[int],
    P: PFunc,
    gamma: float = 0.98,
    tol: float = 1e-8,
    max_outer: int = 1_000,
):
    """Return (policy_fn, V)."""
    raise NotImplementedError
