#######################################################################
# PPO on MountainCar: Comparing Stable-Baselines3 PPO with Tile-Coding SARSA
#
# This script compares a modern policy gradient method (PPO via
# Stable-Baselines3) against the tile-coding semi-gradient SARSA
# from Chapter 10 of Sutton & Barto on the MountainCar problem.
#
# The MountainCar problem requires an under-powered car to build
# momentum by rocking back and forth to reach a goal at the top of
# a hill. This comparison shows how modern deep RL approaches
# stack up against the carefully crafted tile-coding approach from
# the textbook.
#######################################################################

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import floor

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym


# ---------------------------------------------------------------------------
# Tile-Coding SARSA (adapted from Chapter 10)
# ---------------------------------------------------------------------------
class IHT:
    """Structure to handle collisions in tile coding."""
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tile_indices = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tile_indices.append(hash_coords(coords, iht_or_size, read_only))
    return tile_indices


# Mountain Car constants
ACTION_REVERSE = -1
ACTION_ZERO = 0
ACTION_FORWARD = 1
ACTIONS = [ACTION_REVERSE, ACTION_ZERO, ACTION_FORWARD]
POSITION_MIN = -1.2
POSITION_MAX = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07


def mc_step(position, velocity, action):
    """Take one step in the mountain car environment."""
    new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    new_velocity = min(max(VELOCITY_MIN, new_velocity), VELOCITY_MAX)
    new_position = position + new_velocity
    new_position = min(max(POSITION_MIN, new_position), POSITION_MAX)
    reward = -1.0
    if new_position == POSITION_MIN:
        new_velocity = 0.0
    return new_position, new_velocity, reward


class TileCodingValueFunction:
    """State-action value function with tile coding."""
    def __init__(self, step_size, num_of_tilings=8, max_size=2048):
        self.max_size = max_size
        self.num_of_tilings = num_of_tilings
        self.step_size = step_size / num_of_tilings
        self.hash_table = IHT(max_size)
        self.weights = np.zeros(max_size)
        self.position_scale = self.num_of_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.num_of_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    def get_active_tiles(self, position, velocity, action):
        active_tiles = tiles(self.hash_table, self.num_of_tilings,
                            [self.position_scale * position,
                             self.velocity_scale * velocity],
                            [action])
        return active_tiles

    def value(self, position, velocity, action):
        if position == POSITION_MAX:
            return 0.0
        active_tiles = self.get_active_tiles(position, velocity, action)
        return np.sum(self.weights[active_tiles])

    def learn(self, position, velocity, action, target):
        active_tiles = self.get_active_tiles(position, velocity, action)
        estimation = np.sum(self.weights[active_tiles])
        delta = self.step_size * (target - estimation)
        for active_tile in active_tiles:
            self.weights[active_tile] += delta


def get_action(position, velocity, value_function, epsilon=0.0):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.choice(ACTIONS)
    values = [value_function.value(position, velocity, a) for a in ACTIONS]
    max_val = np.max(values)
    best_actions = [a for a, v in zip(ACTIONS, values) if v == max_val]
    return np.random.choice(best_actions)


def semi_gradient_sarsa_episode(value_function, max_steps=1000):
    """Run one episode of semi-gradient SARSA and return the number of steps."""
    position = np.random.uniform(-0.6, -0.4)
    velocity = 0.0
    action = get_action(position, velocity, value_function)
    gamma = 1.0

    for step_count in range(1, max_steps + 1):
        new_position, new_velocity, reward = mc_step(position, velocity, action)

        if new_position == POSITION_MAX:
            value_function.learn(position, velocity, action, reward)
            return step_count

        new_action = get_action(new_position, new_velocity, value_function)
        target = reward + gamma * value_function.value(new_position, new_velocity, new_action)
        value_function.learn(position, velocity, action, target)

        position = new_position
        velocity = new_velocity
        action = new_action

    return max_steps


def run_tile_coding_sarsa(episodes=200, alpha=0.3, num_tilings=8, runs=1):
    """Run tile-coding SARSA and return steps per episode, averaged over runs."""
    all_steps = np.zeros(episodes)
    for run in range(runs):
        vf = TileCodingValueFunction(alpha, num_tilings)
        for ep in range(episodes):
            steps = semi_gradient_sarsa_episode(vf)
            all_steps[ep] += steps
    all_steps /= runs
    return all_steps


# ---------------------------------------------------------------------------
# PPO with Stable-Baselines3
# ---------------------------------------------------------------------------
def run_ppo_mountain_car(total_timesteps=50000, episodes_to_eval=200):
    """Train PPO on MountainCar-v0 and return steps per episode during evaluation."""
    env = gym.make('MountainCar-v0')

    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )
    model.learn(total_timesteps=total_timesteps)

    # Evaluate: record steps per episode
    steps_per_episode = []
    for _ in range(episodes_to_eval):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        steps_per_episode.append(steps)

    env.close()
    return steps_per_episode


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def run_comparison(sarsa_episodes=200, ppo_timesteps=50000, runs=3, save_path=None):
    """Compare tile-coding SARSA and PPO on MountainCar."""
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'images', 'ppo_vs_sarsa_mountain_car.png'
        )

    print("Running tile-coding SARSA...")
    sarsa_steps = run_tile_coding_sarsa(
        episodes=sarsa_episodes, alpha=0.3, num_tilings=8, runs=runs
    )

    print("Training PPO...")
    ppo_steps = run_ppo_mountain_car(
        total_timesteps=ppo_timesteps, episodes_to_eval=sarsa_episodes
    )
    ppo_steps = np.array(ppo_steps, dtype=np.float64)

    # Plot comparison
    plt.figure(figsize=(10, 6))

    # SARSA learning curve (steps during training)
    plt.subplot(1, 2, 1)
    plt.plot(sarsa_steps, label='Tile-Coding SARSA', color='blue', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Steps per Episode')
    plt.title('SARSA Learning Curve (Training)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PPO evaluation performance
    plt.subplot(1, 2, 2)
    window = 10
    ppo_smooth = np.convolve(ppo_steps, np.ones(window)/window, mode='valid')
    plt.plot(ppo_smooth, label='PPO (post-training)', color='orange', alpha=0.8)
    plt.axhline(y=np.mean(ppo_steps), color='red', linestyle='--',
                label=f'PPO mean: {np.mean(ppo_steps):.0f}', alpha=0.7)
    plt.axhline(y=np.mean(sarsa_steps[-50:]), color='blue', linestyle='--',
                label=f'SARSA final mean: {np.mean(sarsa_steps[-50:]):.0f}', alpha=0.7)
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Steps per Episode')
    plt.title('PPO Evaluation vs SARSA Final Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('MountainCar: Tile-Coding SARSA (Ch.10) vs PPO (SB3)', fontsize=14)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

    return sarsa_steps, ppo_steps


def run_quick_test(episodes=3, timesteps=256):
    """Quick test to verify both methods run without errors."""
    sarsa_steps = run_tile_coding_sarsa(episodes=episodes, runs=1)
    assert len(sarsa_steps) == episodes

    # Quick PPO test
    env = gym.make('MountainCar-v0')
    model = PPO('MlpPolicy', env, n_steps=128, batch_size=64, verbose=0)
    model.learn(total_timesteps=timesteps)
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    env.close()
    print("Quick test passed.")
    return sarsa_steps


if __name__ == '__main__':
    run_comparison(sarsa_episodes=200, ppo_timesteps=50000, runs=3)
