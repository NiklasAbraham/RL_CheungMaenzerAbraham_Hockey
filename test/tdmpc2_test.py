"""
Comprehensive test suite for TD-MPC2 implementation
Because the paper is very fucking complicated and I want to make sure I'm not doing anything wrong
Checking all the stuff i atleast can check, mostly dim and shape of the tensors


Niklas
"""

import sys
import unittest


class TestUtils(unittest.TestCase):
    """Test the utility functions."""

    def test_simnorm(self):
        """Test the SimNorm layer."""
        import torch

        from rl_hockey.TD_MPC2.util import SimNorm

        batch_size = 1
        feature_dim = 10
        simplex_dim = 5

        simnorm = SimNorm(feature_dim, simplex_dim)

        x = torch.randn(batch_size, feature_dim)
        print(f"Input shape: {x.shape}")
        print(f"Input: {x}")
        y = simnorm(x)
        print(f"Output shape: {y.shape}")
        print(f"Output: {y}")
        self.assertEqual(y.shape, (batch_size, feature_dim))

    def test_fail_simnorm(self):
        """Test the SimNorm layer with invalid group reshape input."""
        import torch

        from rl_hockey.TD_MPC2.util import SimNorm

        batch_size = 1
        feature_dim = 10
        simplex_dim = 6

        simnorm = SimNorm(feature_dim, simplex_dim)

        x = torch.randn(batch_size, feature_dim)
        print(f"Input shape: {x.shape}")
        print(f"Input: {x}")

        with self.assertRaises(ValueError):
            _ = simnorm(x)


def run_all_tests():
    """Run all test suites and print a summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestUtils,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )
    print("=" * 60)

    if result.wasSuccessful():
        print("\n✓ All tests passed! Implementation tested and verified.")
    else:
        print("\n✗ Some tests failed. Please check the output above for details.")

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("TD-MPC2 Implementation Test")
    print("=" * 60)
    print()

    success = run_all_tests()
    sys.exit(0 if success else 1)
