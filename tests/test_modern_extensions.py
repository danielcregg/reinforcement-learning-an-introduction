#######################################################################
# Test Modern Extensions: Verify DQN, PPO, and Gymnasium scripts run
#
# These tests run short versions of each modern extension script to
# verify they execute without errors. Full training runs are too slow
# for a test suite, so we use minimal episode counts.
#######################################################################

import sys
import os
import pytest

# Add the repo root to the path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class TestDQNCliffWalking:
    """Tests for the DQN cliff walking comparison script."""

    def test_tabular_q_learning_runs(self):
        """Test that tabular Q-learning produces results for a few episodes."""
        from modern_extensions.dqn_cliff_walking import tabular_q_learning
        rewards = tabular_q_learning(episodes=5)
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_dqn_runs(self):
        """Test that DQN produces results for a few episodes."""
        from modern_extensions.dqn_cliff_walking import dqn_cliff_walking
        rewards = dqn_cliff_walking(episodes=5)
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)

    def test_environment_step(self):
        """Test that the environment step function works correctly."""
        from modern_extensions.dqn_cliff_walking import env_step, START, GOAL
        # From start, going up should work
        next_state, reward = env_step(START, 0)  # ACTION_UP
        assert reward == -1
        assert next_state[0] == 2  # moved up one row

    def test_quick_test_function(self):
        """Test the built-in quick test runs without errors."""
        from modern_extensions.dqn_cliff_walking import run_quick_test
        tab_r, dqn_r = run_quick_test(episodes=3)
        assert len(tab_r) == 3
        assert len(dqn_r) == 3


class TestPPOMountainCar:
    """Tests for the PPO mountain car comparison script."""

    def test_tile_coding_sarsa_runs(self):
        """Test that tile-coding SARSA runs for a few episodes."""
        from modern_extensions.ppo_mountain_car import run_tile_coding_sarsa
        steps = run_tile_coding_sarsa(episodes=3, runs=1)
        assert len(steps) == 3
        assert all(s > 0 for s in steps)

    def test_ppo_quick_test(self):
        """Test that PPO can be initialized and trained briefly."""
        from modern_extensions.ppo_mountain_car import run_quick_test
        result = run_quick_test(episodes=2, timesteps=256)
        assert len(result) == 2


class TestGymnasiumComparison:
    """Tests for the Gymnasium comparison script."""

    def test_cliff_walking_gymnasium(self):
        """Test Q-learning on Gymnasium CliffWalking-v1."""
        from modern_extensions.gymnasium_comparison import tabular_q_learning
        import gymnasium as gym
        env = gym.make('CliffWalking-v1')
        q_table, rewards, steps = tabular_q_learning(env, episodes=5)
        assert len(rewards) == 5
        assert q_table.shape[0] == env.observation_space.n
        assert q_table.shape[1] == env.action_space.n
        env.close()

    def test_frozen_lake_gymnasium(self):
        """Test Q-learning on Gymnasium FrozenLake-v1."""
        from modern_extensions.gymnasium_comparison import tabular_q_learning
        import gymnasium as gym
        env = gym.make('FrozenLake-v1')
        q_table, rewards, steps = tabular_q_learning(env, episodes=5)
        assert len(rewards) == 5
        assert q_table.shape == (env.observation_space.n, env.action_space.n)
        env.close()

    def test_quick_test_function(self):
        """Test the built-in quick test runs without errors."""
        from modern_extensions.gymnasium_comparison import run_quick_test
        run_quick_test(episodes=3)
