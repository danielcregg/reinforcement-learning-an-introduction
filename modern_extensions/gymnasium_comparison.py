#######################################################################
# Gymnasium Comparison: Tabular Q-Learning on Standard Benchmarks
#
# This script runs tabular Q-learning (as described in Chapter 6 of
# Sutton & Barto) on Gymnasium's standard environments:
#   - CliffWalking-v1
#   - FrozenLake-v1 (both deterministic and stochastic)
#
# This demonstrates how the textbook algorithms apply directly to
# the standard RL benchmarks used by the community.
#######################################################################

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gymnasium as gym


def tabular_q_learning(env, episodes=500, alpha=0.1, gamma=0.99,
                        epsilon_start=1.0, epsilon_end=0.01,
                        epsilon_decay=0.995):
    """
    Tabular Q-learning on a Gymnasium discrete environment.

    Args:
        env: Gymnasium environment with Discrete observation and action spaces.
        episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay factor per episode.

    Returns:
        q_table: Learned Q-values (n_states x n_actions).
        rewards_per_episode: List of total rewards for each episode.
        steps_per_episode: List of steps taken in each episode.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    steps_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            # Q-learning update
            best_next = np.max(q_table[next_state]) if not terminated else 0.0
            q_table[state, action] += alpha * (
                reward + gamma * best_next - q_table[state, action]
            )

            state = next_state

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return q_table, rewards_per_episode, steps_per_episode


def evaluate_policy(env, q_table, episodes=100):
    """Evaluate a greedy policy derived from a Q-table."""
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def run_cliff_walking(episodes=500, runs=10):
    """Run Q-learning on CliffWalking-v1."""
    all_rewards = np.zeros(episodes)

    for run in range(runs):
        env = gym.make('CliffWalking-v1')
        _, rewards, _ = tabular_q_learning(
            env, episodes=episodes, alpha=0.5, gamma=1.0,
            epsilon_start=0.1, epsilon_end=0.1, epsilon_decay=1.0
        )
        all_rewards += np.array(rewards)
        env.close()

    all_rewards /= runs
    return all_rewards


def run_frozen_lake(episodes=2000, runs=10, is_slippery=True):
    """Run Q-learning on FrozenLake-v1."""
    all_rewards = np.zeros(episodes)

    for run in range(runs):
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
        _, rewards, _ = tabular_q_learning(
            env, episodes=episodes, alpha=0.1, gamma=0.99,
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999
        )
        all_rewards += np.array(rewards)
        env.close()

    all_rewards /= runs
    return all_rewards


def run_comparison(save_path=None):
    """Run Q-learning on both environments and generate comparison plots."""
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'images', 'gymnasium_tabular_qlearning.png'
        )

    print("Running Q-learning on CliffWalking-v1...")
    cliff_rewards = run_cliff_walking(episodes=500, runs=10)

    print("Running Q-learning on FrozenLake-v1 (slippery)...")
    frozen_slippery = run_frozen_lake(episodes=2000, runs=10, is_slippery=True)

    print("Running Q-learning on FrozenLake-v1 (deterministic)...")
    frozen_det = run_frozen_lake(episodes=2000, runs=10, is_slippery=False)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CliffWalking
    window = 20
    cliff_smooth = np.convolve(cliff_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(cliff_smooth, color='blue')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Sum of Rewards')
    axes[0].set_title('CliffWalking-v1\n(Tabular Q-Learning)')
    axes[0].set_ylim([-200, 0])
    axes[0].grid(True, alpha=0.3)

    # FrozenLake (slippery)
    window = 100
    frozen_s_smooth = np.convolve(frozen_slippery, np.ones(window)/window, mode='valid')
    axes[1].plot(frozen_s_smooth, color='green')
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('FrozenLake-v1 (Slippery)\n(Tabular Q-Learning)')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    # FrozenLake (deterministic)
    frozen_d_smooth = np.convolve(frozen_det, np.ones(window)/window, mode='valid')
    axes[2].plot(frozen_d_smooth, color='orange')
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Average Reward')
    axes[2].set_title('FrozenLake-v1 (Deterministic)\n(Tabular Q-Learning)')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Textbook Q-Learning on Gymnasium Benchmarks', fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

    # Print final performance
    print(f"\nFinal Performance (last 50 episodes):")
    print(f"  CliffWalking-v1: {np.mean(cliff_rewards[-50:]):.1f} avg reward")
    print(f"  FrozenLake (slippery): {np.mean(frozen_slippery[-100:]):.3f} success rate")
    print(f"  FrozenLake (deterministic): {np.mean(frozen_det[-100:]):.3f} success rate")


def run_quick_test(episodes=5):
    """Quick test to verify the script runs without errors."""
    env = gym.make('CliffWalking-v1')
    q_table, rewards, steps = tabular_q_learning(env, episodes=episodes)
    assert len(rewards) == episodes
    assert q_table.shape == (env.observation_space.n, env.action_space.n)
    env.close()

    env = gym.make('FrozenLake-v1')
    q_table, rewards, steps = tabular_q_learning(env, episodes=episodes)
    assert len(rewards) == episodes
    env.close()

    print("Quick test passed.")


if __name__ == '__main__':
    run_comparison()
