"""Transition model P(s'|s,a) for DP.

For each (s,a), return a list of (prob, s_next, reward, done).
"""

from typing import List, Tuple

Transition = Tuple[float, int, float, bool]

def transitions(s: int, a: int) -> List[Transition]:
    raise NotImplementedError
