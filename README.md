# Reinforcement Learning: An Introduction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/danielcregg/reinforcement-learning-an-introduction?style=flat-square)

> **Note:** This is a fork of [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction).

## Overview

Python implementations of algorithms from Sutton and Barto's *Reinforcement Learning: An Introduction (2nd Edition)*. Each chapter includes runnable code that reproduces the figures and examples from the textbook.

## Features

- Complete Python implementations for all chapters (1-13)
- Reproduces key figures from the textbook
- Covers multi-armed bandits, dynamic programming, Monte Carlo methods, temporal-difference learning, and policy gradient methods
- Clean, well-documented code suitable for learning and experimentation
- **Modern Extensions** bridging textbook algorithms with deep RL (DQN, PPO) and Gymnasium benchmarks

## Prerequisites

- [Python](https://www.python.org/) 3.8 or higher
- [pip](https://pip.pypa.io/) package manager
- [PyTorch](https://pytorch.org/) (for modern extensions)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/danielcregg/reinforcement-learning-an-introduction.git

# Navigate to the project directory
cd reinforcement-learning-an-introduction

# Install dependencies
pip install -r requirements.txt

# Run an example (e.g., Chapter 2 - Multi-Armed Bandits)
cd chapter02
python ten_armed_testbed.py
```

## Modern Extensions

The `modern_extensions/` directory contains scripts that bridge the textbook's tabular methods with modern deep RL approaches and standard Gymnasium benchmarks.

| Script | Description |
|--------|-------------|
| `dqn_cliff_walking.py` | Deep Q-Network vs tabular Q-learning on CliffWalking (Chapter 6) |
| `ppo_mountain_car.py` | Stable-Baselines3 PPO vs tile-coding SARSA on MountainCar (Chapter 10) |
| `gymnasium_comparison.py` | Tabular Q-learning on Gymnasium's CliffWalking-v1 and FrozenLake-v1 |

### Running the extensions

```bash
# DQN vs Tabular Q-Learning comparison
python modern_extensions/dqn_cliff_walking.py

# PPO vs Tile-Coding SARSA comparison
python modern_extensions/ppo_mountain_car.py

# Tabular Q-Learning on Gymnasium benchmarks
python modern_extensions/gymnasium_comparison.py
```

Comparison plots are saved to the `images/` directory.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Test that all chapter files compile
pytest tests/test_compilation.py -v

# Test modern extensions
pytest tests/test_modern_extensions.py -v
```

## Tech Stack

- **Language:** Python
- **Computation:** NumPy, PyTorch
- **Deep RL:** Stable-Baselines3
- **Environments:** Gymnasium
- **Visualization:** Matplotlib, Seaborn
- **Domain:** Reinforcement Learning

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
