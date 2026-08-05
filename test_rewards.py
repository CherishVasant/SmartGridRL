# test_rewards.py
# Unit tests to verify the correctness of static and dynamic reward shaping formulas.

import unittest
import numpy as np
import rewards

class MockBuilding:
    """
    Mock class to simulate a CityLearn building's net consumption, cost, and emission history.
    """
    def __init__(self, cost_history, emission_history):
        self.net_electricity_consumption_cost = cost_history
        self.net_electricity_consumption_emission = emission_history

class MockCarbonIntensity:
    """
    Mock class to simulate the carbon intensity time-series data container.
    """
    def __init__(self, intensities):
        self._carbon_intensity = intensities

class MockEnvironment:
    """
    Mock class to simulate a CityLearnEnv environment instance.
    """
    def __init__(self, cost_histories, emission_histories, intensities, time_step=1):
        self.buildings = [MockBuilding(c, e) for c, e in zip(cost_histories, emission_histories)]
        # In CityLearn, carbon_intensity lives on the building objects.
        # We assign it to building 0 as expected by our rewards.py.
        self.buildings[0].carbon_intensity = MockCarbonIntensity(intensities)
        self.time_step = time_step

class TestRewardShaping(unittest.TestCase):
    
    def setUp(self):
        # Create dummy cost and emission data for 2 buildings
        # The step completed is at index -1
        self.cost_histories = [[1.2, 0.5], [0.8, 0.3]]  # Current step values: 0.5, 0.3 (Total: 0.8)
        self.emission_histories = [[1.0, 0.4], [0.6, 0.2]]  # Current step values: 0.4, 0.2 (Total: 0.6)
        
        # Carbon intensity time-series: 10 steps
        self.intensities = [0.10, 0.15, 0.20, 0.25, 0.30, 0.10, 0.10, 0.10, 0.10, 0.10]
        
    def test_step_metrics_extraction(self):
        """
        Verify that rewards.get_step_metrics accurately extracts and sums values from index -1.
        """
        env = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities)
        cost, carbon = rewards.get_step_metrics(env)
        self.assertEqual(cost, 0.8)
        self.assertAlmostEqual(carbon, 0.6)
        
    def test_carbon_intensity_extraction(self):
        """
        Verify that get_carbon_intensity retrieves the intensity corresponding to the step that just completed (time_step - 1).
        """
        # Step completed = index 0 (time_step = 1)
        env = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities, time_step=1)
        intensity = rewards.get_carbon_intensity(env)
        self.assertEqual(intensity, 0.10)
        
        # Step completed = index 3 (time_step = 4)
        env2 = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities, time_step=4)
        intensity2 = rewards.get_carbon_intensity(env2)
        self.assertEqual(intensity2, 0.25)

    def test_cost_only_reward(self):
        """
        Verify Cost-Only reward formulation: R = -cost
        """
        env = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities)
        r = rewards.cost_only_reward(env, None)
        self.assertEqual(r, -0.8)

    def test_static_carbon_reward(self):
        """
        Verify Static Carbon-Aware reward formulation: R = -cost - lambda * carbon
        """
        env = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities)
        lambda_val = 2.5
        r = rewards.static_carbon_reward(env, None, lambda_val)
        # Expected: -0.8 - (2.5 * 0.6) = -0.8 - 1.5 = -2.3
        self.assertAlmostEqual(r, -2.3)

    def test_dynamic_carbon_reward_normalization_and_reduction(self):
        """
        Verify that dynamic_carbon_reward:
        1. Correctly maps carbon intensity values in range [carbon_min, carbon_max] to normalized values.
        2. Correctly clips out-of-bounds intensity values.
        3. Mathematically reduces to the static carbon reward when carbon intensity is flat at its maximum.
        """
        carbon_min = 0.10
        carbon_max = 0.30
        lambda_base = 2.0
        
        # Scenario A: Intensity is at max value (0.30)
        # time_step = 5 completed -> intensity = index 4 which is 0.30
        # norm_intensity = (0.30 - 0.10) / (0.30 - 0.10) = 1.0 -> lambda_dynamic = lambda_base = 2.0
        # Expected: -0.8 - (2.0 * 0.6) = -2.0
        env_max = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities, time_step=5)
        r_max = rewards.dynamic_carbon_reward(env_max, None, lambda_base, carbon_min, carbon_max)
        r_static = rewards.static_carbon_reward(env_max, None, lambda_base)
        self.assertAlmostEqual(r_max, -2.0)
        self.assertEqual(r_max, r_static)  # Dynamic reduces to static at max intensity (flat 1.0)
        
        # Scenario B: Intensity is at min value (0.10)
        # time_step = 1 completed -> intensity = index 0 which is 0.10
        # norm_intensity = 0.0 -> lambda_dynamic = 0.0
        # Expected: -0.8 - (0.0 * 0.6) = -0.8 (reduces to cost-only)
        env_min = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities, time_step=1)
        r_min = rewards.dynamic_carbon_reward(env_min, None, lambda_base, carbon_min, carbon_max)
        self.assertAlmostEqual(r_min, -0.8)
        
        # Scenario C: Intensity is at midpoint (0.20)
        # time_step = 3 completed -> intensity = index 2 which is 0.20
        # norm_intensity = 0.5 -> lambda_dynamic = 1.0
        # Expected: -0.8 - (1.0 * 0.6) = -1.4
        env_mid = MockEnvironment(self.cost_histories, self.emission_histories, self.intensities, time_step=3)
        r_mid = rewards.dynamic_carbon_reward(env_mid, None, lambda_base, carbon_min, carbon_max)
        self.assertAlmostEqual(r_mid, -1.4)
        
        # Scenario D: Intensity is out-of-bounds high (0.35) -> should clip to max
        intensities_high = [0.35]
        env_high = MockEnvironment(self.cost_histories, self.emission_histories, intensities_high, time_step=1)
        r_high = rewards.dynamic_carbon_reward(env_high, None, lambda_base, carbon_min, carbon_max)
        self.assertAlmostEqual(r_high, -2.0)  # Clipped to 1.0 -> same as max intensity
        
        # Scenario E: Intensity is out-of-bounds low (0.05) -> should clip to min
        intensities_low = [0.05]
        env_low = MockEnvironment(self.cost_histories, self.emission_histories, intensities_low, time_step=1)
        r_low = rewards.dynamic_carbon_reward(env_low, None, lambda_base, carbon_min, carbon_max)
        self.assertAlmostEqual(r_low, -0.8)  # Clipped to 0.0 -> same as min intensity

if __name__ == '__main__':
    unittest.main()
