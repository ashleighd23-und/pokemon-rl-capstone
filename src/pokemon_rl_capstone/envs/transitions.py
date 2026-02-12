"""Explicit transition model P(s'|s,a) for DP.

models a single-agent MDP by baking in the opponent as a fixed (random legal) policy.

Turn order:
  1) Agent acts (move or switch)
  2) If opponent still has a living Pokemon, opponent acts (random legal)

Stochasticity:
  - Move accuracy: hit with probability ACC else miss (0 damage)
No crits, no randomness beyond accuracy (for V1).

Actions:
  0 = move_0 (power=1)
  1 = move_1 (power=2)
  2 = switch (swap active <-> bench)

Reward:
  +1 if agent wins in resulting terminal
  -1 if agent loses in resulting terminal
   0 otherwise
"""

from __future__ import annotations
from typing import List, Tuple
from .state_space import (
    encode_state, decode_state, is_terminal_state,
    legal_actions_for_agent, legal_actions_for_opp,
    A_T, A_HP, AB_T, AB_HP, O_T, O_HP, OB_T, OB_HP,
)

Transition = Tuple[float, int, float, bool]

ACC = 0.9

# type advantage: Fire>Grass, Grass>Water, Water>Fire
# returns 2 if attacker has advantage else 1
def type_multiplier(att_t: int, def_t: int) -> int:
    if (att_t == 0 and def_t == 2) or (att_t == 2 and def_t == 1) or (att_t == 1 and def_t == 0):
        return 2
    return 1

def apply_damage(hp: int, dmg: int) -> int:
    return max(0, hp - dmg)

def auto_switch_if_needed(s: list[int], *, is_opp: bool) -> None:
    """If active fainted but bench alive, auto-switch to bench."""
    if not is_opp:
        if s[A_HP] == 0 and s[AB_HP] > 0:
            s[A_T], s[AB_T] = s[AB_T], s[A_T]
            s[A_HP], s[AB_HP] = s[AB_HP], s[A_HP]
    else:
        if s[O_HP] == 0 and s[OB_HP] > 0:
            s[O_T], s[OB_T] = s[OB_T], s[O_T]
            s[O_HP], s[OB_HP] = s[OB_HP], s[O_HP]

def do_action_agent(s: list[int], a: int, *, hit: bool) -> None:
    if a == 2:
        # switch
        if s[AB_HP] > 0 and s[A_HP] > 0:
            s[A_T], s[AB_T] = s[AB_T], s[A_T]
            s[A_HP], s[AB_HP] = s[AB_HP], s[A_HP]
        return

    # moves
    if s[A_HP] == 0:
        return
    power = 1 if a == 0 else 2
    dmg = power * type_multiplier(s[A_T], s[O_T]) if hit else 0
    s[O_HP] = apply_damage(s[O_HP], dmg)
    auto_switch_if_needed(s, is_opp=True)

def do_action_opp(s: list[int], a: int, *, hit: bool) -> None:
    if a == 2:
        if s[OB_HP] > 0 and s[O_HP] > 0:
            s[O_T], s[OB_T] = s[OB_T], s[O_T]
            s[O_HP], s[OB_HP] = s[OB_HP], s[O_HP]
        return

    if s[O_HP] == 0:
        return
    power = 1 if a == 0 else 2
    dmg = power * type_multiplier(s[O_T], s[A_T]) if hit else 0
    s[A_HP] = apply_damage(s[A_HP], dmg)
    auto_switch_if_needed(s, is_opp=False)

def terminal_reward(s_tuple) -> float:
    # +1 if opponent dead; -1 if agent dead
    agent_dead = (s_tuple[A_HP] == 0 and s_tuple[AB_HP] == 0)
    opp_dead   = (s_tuple[O_HP] == 0 and s_tuple[OB_HP] == 0)
    if opp_dead and not agent_dead:
        return 1.0
    if agent_dead and not opp_dead:
        return -1.0
    return 0.0

def transitions(s_id: int, a: int) -> List[Transition]:
    s = list(decode_state(s_id))

    # Terminal is absorbing
    if is_terminal_state(tuple(s)):
        return [(1.0, s_id, 0.0, True)]

    # Illegal actions should not be chosen by the policy; treat as no-op with penalty-free transition
    if a not in legal_actions_for_agent(tuple(s)):
        return [(1.0, s_id, 0.0, False)]

    out: List[Transition] = []

    # Agent hit/miss branches (switch is deterministic)
    agent_hit_branches = [(1.0, True)] if a == 2 else [(ACC, True), (1.0 - ACC, False)]

    for p_hit_a, hit_a in agent_hit_branches:
        s1 = s.copy()
        do_action_agent(s1, a, hit=hit_a)

        # If game ended after agent action, stop
        if is_terminal_state(tuple(s1)):
            r = terminal_reward(tuple(s1))
            out.append((p_hit_a, encode_state(tuple(s1)), r, True))
            continue

        # Opponent chooses random legal action uniformly
        opp_acts = legal_actions_for_opp(tuple(s1))
        p_choice = 1.0 / len(opp_acts)

        for a_opp in opp_acts:
            # Opp hit/miss (switch deterministic)
            opp_hit_branches = [(1.0, True)] if a_opp == 2 else [(ACC, True), (1.0 - ACC, False)]
            for p_hit_o, hit_o in opp_hit_branches:
                s2 = s1.copy()
                do_action_opp(s2, a_opp, hit=hit_o)

                done = is_terminal_state(tuple(s2))
                r = terminal_reward(tuple(s2)) if done else 0.0
                out.append((p_hit_a * p_choice * p_hit_o, encode_state(tuple(s2)), r, done))

    # Combine identical next-states (nice for DP stability)
    merged = {}
    for p, ns, r, done in out:
        key = (ns, r, done)
        merged[key] = merged.get(key, 0.0) + p

    return [(p, ns, r, done) for (ns, r, done), p in merged.items()]
