"""Policy iteration (policy evaluation + greedy improvement)."""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple

from .policy_evaluation import policy_evaluation, PFunc
from ..envs.state_space import decode_state, legal_actions_for_agent, is_terminal_state

PolicyTable = Dict[int, Dict[int, float]]

def _uniform_policy_for_state(s_id: int) -> Dict[int, float]:
    s = decode_state(s_id)
    if is_terminal_state(s):
        return {}
    acts = legal_actions_for_agent(s)
    p = 1.0 / len(acts)
    return {a: p for a in acts}

def policy_iteration(
    states: list[int],
    actions: list[int],
    P: PFunc,
    gamma: float = 0.98,
    tol: float = 1e-10,
    max_outer: int = 1_000,
):
    # Initialize uniform random over legal actions
    policy_table: PolicyTable = {s: _uniform_policy_for_state(s) for s in states}

    def policy_fn(s: int) -> Dict[int, float]:
        return policy_table.get(s, {})

    for _ in range(max_outer):
        V = policy_evaluation(states, actions, P, policy_fn, gamma=gamma, tol=tol)

        stable = True
        for s in states:
            s_tuple = decode_state(s)
            if is_terminal_state(s_tuple):
                continue

            legal = legal_actions_for_agent(s_tuple)
            if not legal:
                continue

            # Old best action (if deterministic) or argmax prob
            old_pi = policy_table[s]
            old_best = max(old_pi.items(), key=lambda kv: kv[1])[0] if old_pi else legal[0]

            # Compute greedy action via Q(s,a) from V
            best_a = None
            best_q = None
            for a in legal:
                q = 0.0
                for p, s2, r, done in P(s, a):
                    q += p * (r + (0.0 if done else gamma * V[s2]))
                if (best_q is None) or (q > best_q):
                    best_q = q
                    best_a = a

            # Update deterministic policy
            policy_table[s] = {best_a: 1.0}
            if best_a != old_best:
                stable = False

        if stable:
            break

    return policy_fn, V
