"""Fixed opponent policies for MDP wrapper (V1)."""

import random
from typing import List

def random_policy(legal_actions: List[int]) -> int:
    return random.choice(legal_actions)
