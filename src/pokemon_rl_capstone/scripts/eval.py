"""Eval entrypoint: load saved deterministic policy and estimate win-rate by simulation."""

from __future__ import annotations
import argparse
import os
import pickle
import random

from ..envs.state_space import (
    encode_state, decode_state, is_terminal_state,
    A_HP, AB_HP, O_HP, OB_HP, legal_actions_for_agent
)
from ..envs.transitions import transitions

def sample_next(s_id: int, a: int) -> tuple[int, float, bool]:
    trans = transitions(s_id, a)
    r = random.random()
    c = 0.0
    for p, s2, rew, done in trans:
        c += p
        if r <= c:
            return s2, rew, done
    # numerical fallback
    p, s2, rew, done = trans[-1]
    return s2, rew, done

def random_initial_state() -> int:
    # Simple random start: all mons alive with hp=2
    # random types for active/bench on each side
    import random
    a_t = random.randint(0,2); ab_t = random.randint(0,2)
    o_t = random.randint(0,2); ob_t = random.randint(0,2)
    s = (a_t, 2, ab_t, 2, o_t, 2, ob_t, 2)
    return encode_state(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="artifacts/policy.pkl")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()

    with open(args.policy, "rb") as f:
        policy_table = pickle.load(f)

    wins = 0
    losses = 0
    draws = 0

    for _ in range(args.episodes):
        s = random_initial_state()
        done = False
        steps = 0

        while (not done) and steps < args.max_steps:
            steps += 1
            st = decode_state(s)
            if is_terminal_state(st):
                done = True
                break

            # Choose action from policy if available; else random legal
            if s in policy_table:
                a = policy_table[s]
            else:
                legal = legal_actions_for_agent(st)
                a = random.choice(legal) if legal else 0

            s, r, done = sample_next(s, a)

            if done:
                if r > 0:
                    wins += 1
                elif r < 0:
                    losses += 1
                else:
                    draws += 1

        if not done:
            draws += 1

    total = wins + losses + draws
    print(f"Episodes: {total}")
    print(f"Wins:     {wins}  ({wins/total:.3f})")
    print(f"Losses:   {losses} ({losses/total:.3f})")
    print(f"Draws:    {draws}  ({draws/total:.3f})")

if __name__ == "__main__":
    main()
