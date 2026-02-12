"""Train entrypoint: run DP policy iteration and save artifacts."""

from __future__ import annotations
import argparse
import os
import pickle
import numpy as np

from ..envs.state_space import enumerate_states, encode_state, num_states
from ..envs.transitions import transitions
from ..dp.policy_iteration import policy_iteration

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()

    # Build state id list
    states = [encode_state(s) for s in enumerate_states()]
    actions = [0, 1, 2]

    policy_fn, V = policy_iteration(states, actions, transitions, gamma=args.gamma)

    os.makedirs(args.outdir, exist_ok=True)

    # Save policy as a dict: state_id -> action (deterministic) or probs
    policy_table = {}
    for s in states:
        pi = policy_fn(s)
        if not pi:
            continue
        # deterministic
        a = max(pi.items(), key=lambda kv: kv[1])[0]
        policy_table[s] = a

    with open(os.path.join(args.outdir, "policy.pkl"), "wb") as f:
        pickle.dump(policy_table, f)

    # Save V as dense array for convenience
    V_arr = np.zeros(num_states(), dtype=np.float64)
    for s, v in V.items():
        V_arr[s] = v
    np.save(os.path.join(args.outdir, "V.npy"), V_arr)

    print(f"Saved policy to {args.outdir}/policy.pkl")
    print(f"Saved value function to {args.outdir}/V.npy")

if __name__ == "__main__":
    main()
