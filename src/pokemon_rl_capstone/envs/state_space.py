"""State encoding/decoding + enumeration for the tabular Pokemon micro-battle MDP.

State tuple:
  (a_type, a_hp, ab_type, ab_hp, o_type, o_hp, ob_type, ob_hp)

Types: 0=Fire, 1=Water, 2=Grass
HP bins: 0,1,2   (0 = fainted)
"""

from __future__ import annotations
from typing import Iterable, Tuple

N_TYPES = 3
N_HP = 3

# Tuple indices for readability
A_T, A_HP, AB_T, AB_HP, O_T, O_HP, OB_T, OB_HP = range(8)

RADICES = [N_TYPES, N_HP, N_TYPES, N_HP, N_TYPES, N_HP, N_TYPES, N_HP]

def encode_state(s: Tuple[int, ...]) -> int:
    """Encode an 8-tuple into an integer id using mixed radix."""
    assert len(s) == 8
    sid = 0
    mult = 1
    for val, base in zip(reversed(s), reversed(RADICES)):
        if not (0 <= val < base):
            raise ValueError(f"Value {val} out of range for base {base} in state {s}")
        sid += val * mult
        mult *= base
    return sid

def decode_state(state_id: int) -> Tuple[int, ...]:
    """Decode integer id back into the 8-tuple."""
    if state_id < 0:
        raise ValueError("state_id must be non-negative")
    vals = []
    x = state_id
    for base in reversed(RADICES):
        vals.append(x % base)
        x //= base
    vals.reverse()
    return tuple(vals)

def num_states() -> int:
    n = 1
    for b in RADICES:
        n *= b
    return n

def enumerate_states() -> Iterable[Tuple[int, ...]]:
    """Yield all states (including terminal-like ones where HP==0)."""
    for a_t in range(N_TYPES):
        for a_hp in range(N_HP):
            for ab_t in range(N_TYPES):
                for ab_hp in range(N_HP):
                    for o_t in range(N_TYPES):
                        for o_hp in range(N_HP):
                            for ob_t in range(N_TYPES):
                                for ob_hp in range(N_HP):
                                    yield (a_t, a_hp, ab_t, ab_hp, o_t, o_hp, ob_t, ob_hp)

def is_terminal_state(s: Tuple[int, ...]) -> bool:
    """Terminal if BOTH active and bench are fainted for either player."""
    a_dead = (s[A_HP] == 0 and s[AB_HP] == 0)
    o_dead = (s[O_HP] == 0 and s[OB_HP] == 0)
    return a_dead or o_dead

def legal_actions_for_agent(s: Tuple[int, ...]) -> list[int]:
    """Agent actions: 0=move0, 1=move1, 2=switch (only if bench alive)."""
    if is_terminal_state(s):
        return []
    acts = [0, 1]
    if s[AB_HP] > 0 and s[A_HP] > 0:
        acts.append(2)
    return acts

def legal_actions_for_opp(s: Tuple[int, ...]) -> list[int]:
    """Opponent legal actions from the opponent perspective."""
    if is_terminal_state(s):
        return []
    acts = [0, 1]
    # Opp bench alive and opp active alive
    if s[OB_HP] > 0 and s[O_HP] > 0:
        acts.append(2)
    return acts
