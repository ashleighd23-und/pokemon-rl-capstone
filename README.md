
# Pokémon RL Capstone

Pokemon-inspired tabular Markov Decision Process (MDP) with Dynamic Programming (Policy Iteration) as the foundation for a scalable Reinforcement Learning Capstone Project for EECS590 at the University of North Dakota.

This project satisfies V1 requirements of the EECS590 Capstone and is structured using the Cookiecutter Data Science template.

---

# V1 Requirements Coverage

## 1. GitHub Repository Formatting

This repository follows the Cookiecutter Data Science structure:

- Clean modular `src/` layout
- Clear separation of environment, algorithms, scripts
- Reproducible dependency files
- Dedicated artifacts folder for saved models
- Proper Git history and version control

This structure ensures usability, maintainability, and low deployment risk.

---

## 2. Documentation / Organization

- `README.md` documents environment design and usage
- `requirements.txt` for pip-based environments
- `environment.yml` for conda environments
- Source code located in:
  - `src/pokemon_rl_capstone/envs/`
  - `src/pokemon_rl_capstone/dp/`
  - `src/pokemon_rl_capstone/scripts/`
- Trained artifacts saved to:
  - `artifacts/`

---

# 3. Explicit MDP Representation (Foundation Environment)

The environment is a Pokémon-inspired micro battle formulated as a **finite tabular MDP** by fixing the opponent to a stationary random policy.

## State Representation

Each state is a discrete tuple:

```

(a_type, a_hp, ab_type, ab_hp, o_type, o_hp, ob_type, ob_hp)

```

Where:
- Types: {0 = Fire, 1 = Water, 2 = Grass}
- HP bins: {0, 1, 2} (0 = fainted)
- a_* = agent
- o_* = opponent
- ab / ob = bench Pokémon

States are encoded to integer IDs for tabular DP.

---

## Action Space (Agent)

Actions:
- 0 = move_0 (power = 1)
- 1 = move_1 (power = 2)
- 2 = switch (swap active and bench, if bench alive)

Illegal actions are masked automatically.

---

## Reward Function

- +1 if opponent has no remaining Pokémon
- -1 if agent has no remaining Pokémon
- 0 otherwise

Terminal states are absorbing.

---

## Transition Model

Transitions are explicitly implemented in:

```

src/pokemon_rl_capstone/envs/transitions.py

```

Turn structure:
1. Agent acts
2. Opponent acts (random legal action)

Stochasticity:
- Move accuracy: hit with probability 0.9
- Type advantage multiplier:
  - Fire > Grass
  - Grass > Water
  - Water > Fire

This produces a fully specified tabular transition function:

P(s' | s, a)

This explicit model supports future extensions toward belief-state modeling and world-model learning.

---

# 4. Dynamic Programming Implementation

Implemented algorithms:

- Iterative Policy Evaluation
- Policy Improvement (greedy over Q derived from V)
- Policy Iteration (evaluation + improvement loop)

Located in:

```

src/pokemon_rl_capstone/dp/

```

Policy iteration produces:

- Optimal deterministic policy
- Converged value function V(s)

---

# 5. Agent Framework

Training entrypoint:

```

src/pokemon_rl_capstone/scripts/train.py

```

Evaluation entrypoint:

```

src/pokemon_rl_capstone/scripts/eval.py

```

Saved artifacts:
- artifacts/policy.pkl
- artifacts/V.npy

---

# How To Run (Colab Compatible)

Because this project uses a `src/` layout, use PYTHONPATH:

### Train (Policy Iteration)

```

PYTHONPATH=./src python -m pokemon_rl_capstone.scripts.train --gamma 0.98 --outdir artifacts

```

### Evaluate Policy

```

PYTHONPATH=./src python -m pokemon_rl_capstone.scripts.eval --policy artifacts/policy.pkl --episodes 2000

```

---

# Project Layout

```

src/pokemon_rl_capstone/
envs/
state_space.py
transitions.py
dp/
policy_evaluation.py
policy_iteration.py
agents/
opponent_policies.py
scripts/
train.py
eval.py
utils/
viz.py
artifacts/
notebooks/

```

---

# Roadmap (Next Versions)

Planned additions aligned with course content:

- Value Iteration
- Monte Carlo Control
- Forward-view TD(n)
- Backward-view TD(λ) with eligibility traces
- SARSA(n) and SARSA(λ)
- Q-learning
- Exploration strategies (ε-greedy, softmax)
- Text-based render mode for rollouts
- TensorBoard / Weights & Biases logging

Advanced research direction:

- Two-player Markov game (self-play training)
- League-based opponent pools
- Partial observability + belief-state modeling
- Approximate exploitability via best-response training
- Meta-game payoff matrix analysis

---

# Summary

V1 establishes:

- A fully specified finite MDP
- Explicit stochastic transition dynamics
- Working tabular dynamic programming algorithms
- Modular agent training and evaluation framework

This foundation enables systematic progression from classical DP to deep multi-agent reinforcement learning and game-theoretic analysis.

---

# References

[1] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[2] M. L. Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. New York, NY, USA: John Wiley & Sons, 1994.

[3] M. L. Littman, “Markov games as a framework for multi-agent reinforcement learning,” in *Proceedings of the 11th International Conference on Machine Learning*, 1994, pp. 157–163.

[4] J. Heinrich and D. Silver, “Deep reinforcement learning from self-play in imperfect-information games,” *arXiv preprint arXiv:1603.01121*, 2016.

[5] DrivenData, “Cookiecutter Data Science Template (v1),” [Online]. Available: https://github.com/drivendataorg/cookiecutter-data-science


Important Note: Citations management will be migrated to Zotero for V2 of the project.

---
##Acknowledgments

Portions of repository structuring, documentation drafting, and debugging support were assisted using OpenAI’s ChatGPT (GPT-5 series). All algorithmic implementations, modeling decisions, and validation were independently implemented by the author.
