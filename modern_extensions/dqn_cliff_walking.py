#######################################################################
# DQN on CliffWalking: Comparing Deep Q-Network with Tabular Q-Learning
#
# This script implements a Deep Q-Network (DQN) for the CliffWalking
# problem from Chapter 6 of Sutton & Barto, and compares its learning
# curve against the original tabular Q-learning implementation.
#
# The CliffWalking environment is a 4x12 grid where the agent must
# navigate from start (bottom-left) to goal (bottom-right) while
# avoiding a cliff along the bottom row. This is a good testbed for
# comparing tabular vs. function approximation approaches because
# the state space is small enough for tabular methods to work well,
# letting us see the overhead/tradeoffs of neural network approximation.
#######################################################################

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CliffWalking Environment (matching the textbook's Chapter 6 formulation)
# ---------------------------------------------------------------------------
WORLD_HEIGHT = 4
WORLD_WIDTH = 12
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
START = [3, 0]
GOAL = [3, 11]
NUM_STATES = WORLD_HEIGHT * WORLD_WIDTH
NUM_ACTIONS = len(ACTIONS)


def state_to_index(state):
    """Convert [row, col] state to a single integer index."""
    return state[0] * WORLD_WIDTH + state[1]


def env_step(state, action):
    """Take an action in the cliff walking environment.
    Returns (next_state, reward).
    """
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        raise ValueError(f"Invalid action: {action}")

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward


# ---------------------------------------------------------------------------
# Tabular Q-Learning (from Chapter 6)
# ---------------------------------------------------------------------------
def tabular_q_learning(episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1):
    """Run tabular Q-learning on CliffWalking and return per-episode rewards."""
    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, NUM_ACTIONS))
    rewards_per_episode = []

    for _ in range(episodes):
        state = list(START)
        total_reward = 0.0
        while state != GOAL:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(ACTIONS)
            else:
                values = q_value[state[0], state[1], :]
                action = np.random.choice(
                    [a for a, v in enumerate(values) if v == np.max(values)]
                )

            next_state, reward = env_step(state, action)
            total_reward += reward

            # Q-learning update
            q_value[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(q_value[next_state[0], next_state[1], :])
                - q_value[state[0], state[1], action]
            )
            state = next_state
        rewards_per_episode.append(total_reward)

    return rewards_per_episode


# ---------------------------------------------------------------------------
# DQN Components
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Simple feedforward Q-network for CliffWalking."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Simple experience replay buffer."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def state_to_onehot(state):
    """Convert [row, col] state to one-hot vector."""
    vec = np.zeros(NUM_STATES, dtype=np.float32)
    vec[state_to_index(state)] = 1.0
    return vec


def dqn_cliff_walking(episodes=500, gamma=1.0, epsilon_start=1.0,
                       epsilon_end=0.1, epsilon_decay=0.995,
                       lr=1e-3, batch_size=32, target_update=10,
                       buffer_size=10000, max_steps_per_episode=200):
    """Run DQN on CliffWalking and return per-episode rewards."""
    device = torch.device("cpu")

    policy_net = QNetwork(NUM_STATES, NUM_ACTIONS).to(device)
    target_net = QNetwork(NUM_STATES, NUM_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)
    epsilon = epsilon_start
    rewards_per_episode = []

    for episode in range(episodes):
        state = list(START)
        state_vec = state_to_onehot(state)
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(state_vec).unsqueeze(0).to(device))
                    action = q_vals.argmax(dim=1).item()

            next_state, reward = env_step(state, action)
            next_state_vec = state_to_onehot(next_state)
            done = (next_state == GOAL)
            total_reward += reward

            replay_buffer.push(state_vec, action, reward, next_state_vec, done)

            state = next_state
            state_vec = next_state_vec

            # Train if we have enough samples
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = \
                    replay_buffer.sample(batch_size)

                states_t = torch.FloatTensor(states_b).to(device)
                actions_t = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards_t = torch.FloatTensor(rewards_b).to(device)
                next_states_t = torch.FloatTensor(next_states_b).to(device)
                dones_t = torch.FloatTensor(dones_b).to(device)

                # Current Q values
                current_q = policy_net(states_t).gather(1, actions_t).squeeze()

                # Target Q values
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(dim=1)[0]
                    target_q = rewards_t + gamma * next_q * (1 - dones_t)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards_per_episode.append(total_reward)

    return rewards_per_episode


# ---------------------------------------------------------------------------
# Comparison Plot
# ---------------------------------------------------------------------------
def run_comparison(episodes=500, runs=10, save_path=None):
    """Run both methods multiple times and plot averaged learning curves."""
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'images', 'dqn_vs_tabular_cliff_walking.png'
        )

    tabular_rewards = np.zeros(episodes)
    dqn_rewards = np.zeros(episodes)

    for r in range(runs):
        print(f"Run {r + 1}/{runs}")
        tab_r = tabular_q_learning(episodes=episodes)
        dqn_r = dqn_cliff_walking(episodes=episodes)

        tabular_rewards += np.array(tab_r)
        dqn_rewards += np.array(dqn_r)

    tabular_rewards /= runs
    dqn_rewards /= runs

    # Smooth with a rolling window for readability
    window = 20
    tabular_smooth = np.convolve(tabular_rewards, np.ones(window)/window, mode='valid')
    dqn_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(tabular_smooth, label='Tabular Q-Learning', linewidth=2)
    plt.plot(dqn_smooth, label='DQN', linewidth=2)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards (smoothed)')
    plt.title('CliffWalking: Tabular Q-Learning vs DQN')
    plt.legend()
    plt.ylim([-200, 0])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

    return tabular_rewards, dqn_rewards


def run_quick_test(episodes=5):
    """Quick test to verify both methods run without errors."""
    tab_rewards = tabular_q_learning(episodes=episodes)
    dqn_rewards = dqn_cliff_walking(episodes=episodes)
    assert len(tab_rewards) == episodes
    assert len(dqn_rewards) == episodes
    print("Quick test passed.")
    return tab_rewards, dqn_rewards


if __name__ == '__main__':
    run_comparison(episodes=500, runs=5)
