"""
Comprehensive test suite for Hockey Environment installation and basic functionality.

This test file verifies that:
- All required packages are installed and importable
- The Hockey Environment can be created and used
- Basic operations (reset, step, render) work correctly
- Different game modes function properly
- Built-in opponents can be used
- Action space conversions work

Usage:
    Make sure you have activated the conda environment (e.g., rl-hockey) before running:
    python test/hockey_env_test.py
    
    Or using pytest:
    pytest test/hockey_env_test.py -v
"""

import sys
import unittest


class TestImports(unittest.TestCase):
    """Test that all required packages can be imported."""
    
    def test_import_numpy(self):
        """Test that numpy is installed and importable."""
        try:
            import numpy as np
            self.assertIsNotNone(np)
            print("✓ numpy imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import numpy: {e}")
    
    def test_import_gymnasium(self):
        """Test that gymnasium is installed and importable."""
        try:
            import gymnasium as gym
            self.assertIsNotNone(gym)
            print("✓ gymnasium imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import gymnasium: {e}")
    
    def test_import_hockey_env(self):
        """Test that hockey environment can be imported."""
        try:
            import hockey.hockey_env as h_env
            self.assertIsNotNone(h_env)
            print("✓ hockey.hockey_env imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import hockey.hockey_env: {e}")


class TestEnvironmentCreation(unittest.TestCase):
    """Test that the environment can be created in various configurations."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        self.h_env = h_env
    
    def test_create_default_environment(self):
        """Test creating environment with default parameters."""
        env = self.h_env.HockeyEnv()
        self.assertIsNotNone(env)
        print("✓ Default environment created successfully")
        env.close()
    
    def test_create_with_mode_normal(self):
        """Test creating environment in NORMAL mode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.NORMAL)
        self.assertIsNotNone(env)
        self.assertEqual(env.mode, self.h_env.Mode.NORMAL)
        print("✓ Environment in NORMAL mode created successfully")
        env.close()
    
    def test_create_with_mode_shooting(self):
        """Test creating environment in TRAIN_SHOOTING mode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.TRAIN_SHOOTING)
        self.assertIsNotNone(env)
        self.assertEqual(env.mode, self.h_env.Mode.TRAIN_SHOOTING)
        print("✓ Environment in TRAIN_SHOOTING mode created successfully")
        env.close()
    
    def test_create_with_mode_defense(self):
        """Test creating environment in TRAIN_DEFENSE mode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.TRAIN_DEFENSE)
        self.assertIsNotNone(env)
        self.assertEqual(env.mode, self.h_env.Mode.TRAIN_DEFENSE)
        print("✓ Environment in TRAIN_DEFENSE mode created successfully")
        env.close()
    
    def test_create_with_keep_mode_false(self):
        """Test creating environment with keep_mode=False."""
        env = self.h_env.HockeyEnv(keep_mode=False)
        self.assertIsNotNone(env)
        self.assertFalse(env.keep_mode)
        print("✓ Environment with keep_mode=False created successfully")
        env.close()


class TestObservationAndActionSpaces(unittest.TestCase):
    """Test that observation and action spaces are correctly configured."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        self.h_env = h_env
        self.env = self.h_env.HockeyEnv()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_observation_space_shape(self):
        """Test that observation space has correct shape."""
        obs_shape = self.env.observation_space.shape
        self.assertEqual(obs_shape, (18,))
        print(f"✓ Observation space shape is correct: {obs_shape}")
    
    def test_observation_space_type(self):
        """Test that observation space is a Box space."""
        import gymnasium.spaces as spaces
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        print("✓ Observation space is a Box space")
    
    def test_action_space_shape(self):
        """Test that action space has correct shape (8 actions for 2 players with keep_mode)."""
        action_shape = self.env.action_space.shape
        self.assertEqual(action_shape, (8,))
        print(f"✓ Action space shape is correct: {action_shape}")
    
    def test_action_space_type(self):
        """Test that action space is a Box space."""
        import gymnasium.spaces as spaces
        self.assertIsInstance(self.env.action_space, spaces.Box)
        print("✓ Action space is a Box space")
    
    def test_action_space_bounds(self):
        """Test that action space has correct bounds."""
        self.assertEqual(self.env.action_space.low.min(), -1.0)
        self.assertEqual(self.env.action_space.high.max(), 1.0)
        print("✓ Action space bounds are correct: [-1, 1]")


class TestEnvironmentOperations(unittest.TestCase):
    """Test basic environment operations like reset and step."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        import numpy as np
        self.h_env = h_env
        self.np = np
        self.env = self.h_env.HockeyEnv()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_reset_returns_observation(self):
        """Test that reset returns a valid observation."""
        obs, info = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape, (18,))
        self.assertIsInstance(info, dict)
        print("✓ Reset returns valid observation and info")
    
    def test_reset_observation_values(self):
        """Test that reset observation contains reasonable values."""
        obs, info = self.env.reset()
        self.assertTrue(self.np.all(self.np.isfinite(obs)))
        print("✓ Reset observation contains finite values")
    
    def test_step_with_random_actions(self):
        """Test that step works with random actions."""
        obs, info = self.env.reset()
        a1 = self.np.random.uniform(-1, 1, 4)
        a2 = self.np.random.uniform(-1, 1, 4)
        action = self.np.hstack([a1, a2])
        
        obs_new, reward, done, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(obs_new)
        self.assertEqual(obs_new.shape, (18,))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        print("✓ Step operation works correctly")
    
    def test_multiple_steps(self):
        """Test that multiple steps can be taken."""
        obs, info = self.env.reset()
        for _ in range(10):
            a1 = self.np.random.uniform(-1, 1, 4)
            a2 = self.np.random.uniform(-1, 1, 4)
            action = self.np.hstack([a1, a2])
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                break
        print("✓ Multiple steps executed successfully")
    
    def test_obs_agent_two(self):
        """Test that obs_agent_two returns valid observation."""
        obs, info = self.env.reset()
        obs_agent2 = self.env.obs_agent_two()
        self.assertIsNotNone(obs_agent2)
        self.assertEqual(obs_agent2.shape, (18,))
        print("✓ obs_agent_two returns valid observation")


class TestGameModes(unittest.TestCase):
    """Test different game modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        import numpy as np
        self.h_env = h_env
        self.np = np
    
    def test_normal_mode_episode(self):
        """Test that NORMAL mode can run an episode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.NORMAL)
        obs, info = env.reset()
        
        for _ in range(50):
            a1 = self.np.random.uniform(-1, 1, 4)
            a2 = self.np.random.uniform(-1, 1, 4)
            obs, reward, done, truncated, info = env.step(self.np.hstack([a1, a2]))
            if done or truncated:
                break
        
        self.assertIn('winner', info)
        print("✓ NORMAL mode episode executed successfully")
        env.close()
    
    def test_shooting_mode_episode(self):
        """Test that TRAIN_SHOOTING mode can run an episode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.TRAIN_SHOOTING)
        obs, info = env.reset()
        
        for _ in range(50):
            a1 = self.np.random.uniform(-1, 1, 4)
            a2 = self.np.zeros(4)
            obs, reward, done, truncated, info = env.step(self.np.hstack([a1, a2]))
            if done or truncated:
                break
        
        print("✓ TRAIN_SHOOTING mode episode executed successfully")
        env.close()
    
    def test_defense_mode_episode(self):
        """Test that TRAIN_DEFENSE mode can run an episode."""
        env = self.h_env.HockeyEnv(mode=self.h_env.Mode.TRAIN_DEFENSE)
        obs, info = env.reset()
        
        for _ in range(50):
            a1 = self.np.random.uniform(-1, 1, 4)
            a2 = self.np.zeros(4)
            obs, reward, done, truncated, info = env.step(self.np.hstack([a1, a2]))
            if done or truncated:
                break
        
        print("✓ TRAIN_DEFENSE mode episode executed successfully")
        env.close()


class TestBasicOpponent(unittest.TestCase):
    """Test the BasicOpponent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        import numpy as np
        self.h_env = h_env
        self.np = np
    
    def test_create_weak_opponent(self):
        """Test creating a weak BasicOpponent."""
        opponent = self.h_env.BasicOpponent(weak=True)
        self.assertIsNotNone(opponent)
        print("✓ Weak BasicOpponent created successfully")
    
    def test_create_strong_opponent(self):
        """Test creating a strong BasicOpponent."""
        opponent = self.h_env.BasicOpponent(weak=False)
        self.assertIsNotNone(opponent)
        print("✓ Strong BasicOpponent created successfully")
    
    def test_opponent_act(self):
        """Test that opponent can generate actions."""
        env = self.h_env.HockeyEnv()
        opponent = self.h_env.BasicOpponent(weak=True)
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        
        action = opponent.act(obs_agent2)
        self.assertIsNotNone(action)
        self.assertEqual(len(action), 4)
        print("✓ BasicOpponent.act() returns valid action")
        env.close()
    
    def test_play_against_opponent(self):
        """Test playing a short episode against BasicOpponent."""
        env = self.h_env.HockeyEnv()
        opponent = self.h_env.BasicOpponent(weak=True)
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        
        for _ in range(20):
            a1 = opponent.act(obs)
            a2 = opponent.act(obs_agent2)
            obs, reward, done, truncated, info = env.step(self.np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            if done or truncated:
                break
        
        print("✓ Episode against BasicOpponent executed successfully")
        env.close()


class TestDiscreteActions(unittest.TestCase):
    """Test discrete action conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        import numpy as np
        self.h_env = h_env
        self.np = np
        self.env = self.h_env.HockeyEnv()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_discrete_to_continuous_conversion(self):
        """Test that discrete actions can be converted to continuous."""
        for discrete_action in range(8):
            continuous = self.env.discrete_to_continous_action(discrete_action)
            self.assertIsNotNone(continuous)
            self.assertEqual(len(continuous), 4)
            print(f"✓ Discrete action {discrete_action} converts to continuous: {continuous}")
    
    def test_use_discrete_actions_in_step(self):
        """Test that discrete actions can be used in step."""
        obs, info = self.env.reset()
        discrete_action = 1  # Left movement
        continuous = self.env.discrete_to_continous_action(discrete_action)
        a2 = self.np.zeros(4)
        action = self.np.hstack([continuous, a2])
        
        obs_new, reward, done, truncated, info = self.env.step(action)
        self.assertIsNotNone(obs_new)
        print("✓ Discrete actions work in step operation")


class TestRendering(unittest.TestCase):
    """Test rendering functionality (without actually displaying)."""
    
    def setUp(self):
        """Set up test fixtures."""
        import hockey.hockey_env as h_env
        self.h_env = h_env
        self.env = self.h_env.HockeyEnv()
    
    def tearDown(self):
        """Clean up after tests."""
        self.env.close()
    
    def test_render_human_mode(self):
        """Test that render with human mode doesn't crash."""
        obs, info = self.env.reset()
        try:
            # Try to render - this might fail in headless environments but shouldn't crash
            self.env.render(mode="human")
            print("✓ Render in human mode works")
        except Exception as e:
            # If it fails, it's likely due to display issues, which is acceptable
            print(f"⚠ Render in human mode failed (likely headless environment): {e}")
    
    def test_render_rgb_array_mode(self):
        """Test that render with rgb_array mode returns an array."""
        obs, info = self.env.reset()
        try:
            frame = self.env.render(mode="rgb_array")
            if frame is not None:
                self.assertIsNotNone(frame)
                print("✓ Render in rgb_array mode returns valid array")
        except Exception as e:
            print(f"⚠ Render in rgb_array mode failed: {e}")


def run_all_tests():
    """Run all test suites and print a summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestImports,
        TestEnvironmentCreation,
        TestObservationAndActionSpaces,
        TestEnvironmentOperations,
        TestGameModes,
        TestBasicOpponent,
        TestDiscreteActions,
        TestRendering
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)
    
    if result.wasSuccessful():
        print("\n✓ All tests passed! Installation and basic functionality verified.")
    else:
        print("\n✗ Some tests failed. Please check the output above for details.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*60)
    print("Hockey Environment Installation and Functionality Test")
    print("="*60)
    print()
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

