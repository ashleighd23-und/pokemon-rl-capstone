# Pokemon RL Capstone

Pokemon-inspired tabular MDP + DP (policy iteration) foundation for RL capstone.

## Version 1 Checklist Coverage
This repository is structured using **Cookiecutter Data Science** and formatted to meet V1 requirements of the EECS590 Capstone Project:

Progress so far:

1. Repo formatting (cookiecutter structure + organized modules)
2. Documentation (this README + dependencies)
3. Explicit MDP representation (tabular, DP-friendly)
4. Dynamic Programming (policy iteration + policy improvement planned/implemented)
5. Agent framework (train/eval entrypoints + saved artifacts)


##About the Project


### MDP (Foundation Environment)
The foundation environment is a **Pokemon-inspired micro battle** modeled as an MDP by fixing the opponent to a stationary policy.

- **State:** discrete, encoded to an integer id (types/HP bins/bench flags, etc.)
- **Actions:** small discrete set (moves + optional switch); illegal actions masked
- **Reward:** terminal win/loss (+ optional shaping)
- **Transition:** stochastic outcomes (accuracy/damage variation) + opponent fixed policy

### Project Layout (Cookiecutter DS-compatible)
- `src/pokemon_rl_capstone/envs/` environment, state encoding, transitions
- `src/pokemon_rl_capstone/dp/` dynamic programming algorithms
- `src/pokemon_rl_capstone/agents/` opponent baselines
- `src/pokemon_rl_capstone/scripts/` train/eval entrypoints
- `artifacts/` saved policies and value functions
- `notebooks/` experiments

### Dependencies
- `requirements.txt` (pip)
- `environment.yml` (optional, conda)

### Roadmap (Next Versions)
- Value Iteration
- Monte Carlo methods
- TD(n), TD(lambda)
- SARSA(n), SARSA(lambda)
- Exploration injection + Q-learning
- Later: self-play league training, belief-state inference, exploitability proxy evaluation
