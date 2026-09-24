"""Physics and policy checks, independent of animation and report generation."""

import unittest

import numpy as np

import assignment_2 as experiment
from models import inverted_pendulum_walker as model


class WalkerPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.params = model.generate_params()

    def test_feedback_and_torque_limits(self):
        state = np.array([0.01, -0.005])
        torque = experiment.compute_ankle_torque(state, self.params)
        acceleration = model.dynamics(
            0, state, {**self.params, "ankle_torque": torque}
        )[1]
        self.assertAlmostEqual(acceleration, -state[0] - 2 * state[1])
        states = np.array([[10.0, -10.0], [10.0, -10.0]])
        np.testing.assert_allclose(
            experiment.compute_ankle_torque(states, self.params), [-0.981, 0.4905]
        )

    def test_guard_uses_angle_and_forward_crossing(self):
        touchdown = self.params["incline"] + self.params["angle_of_attack"]
        self.assertTrue(
            model.event_guard([touchdown - 0.01, 2], [touchdown, 2], self.params)
        )
        self.assertFalse(model.event_guard([0, 4], [0.01, 4], self.params))
        self.assertFalse(
            model.event_guard(
                [touchdown + 0.01, -2], [touchdown - 0.01, -2], self.params
            )
        )

    def test_passive_energy_conservation(self):
        state = np.array([0.0, 0.5])
        energy = sum(model.calculate_energy(state, self.params))
        for _ in range(100):
            state = experiment.integrate(state, self.params, 0.001)
        self.assertAlmostEqual(
            sum(model.calculate_energy(state, self.params)), energy, places=10
        )

    def test_impact_geometry_and_energy_loss(self):
        alpha = self.params["angle_of_attack"]
        gamma = self.params["incline"]
        before = np.array([gamma + alpha, 2.0])
        after = model.event_dynamics(before, self.params)
        self.assertAlmostEqual(after[0], gamma - alpha)
        before_ke, _ = model.calculate_energy(before, self.params)
        after_ke, _ = model.calculate_energy(after, self.params)
        self.assertAlmostEqual(after_ke / before_ke, np.cos(2 * alpha) ** 2)

    def test_return_map_matches_energy_solution(self):
        empty_roa = experiment.RegionOfAttraction(
            np.array([-1, 1]), np.array([-5, 5]), np.zeros((2, 2), dtype=bool)
        )
        velocities = np.array([0.8, 1.5, 3.0, 4.0])
        actions = np.linspace(*experiment.ALPHA_BOUNDS, 5)
        gamma = self.params["incline"]
        gravity = self.params["gravity"] / self.params["length"]
        impact_squared = velocities[:, None] ** 2 + 2 * gravity * (
            1 - np.cos(gamma + actions)
        )
        return_squared = np.cos(2 * actions) ** 2 * impact_squared + 2 * gravity * (
            np.cos(gamma - actions) - 1
        )
        expected = np.full(return_squared.shape, np.nan)
        returning = return_squared > 0
        expected[returning] = np.sqrt(return_squared[returning])
        coarse, terminal = experiment.compute_return_map(
            velocities, actions, self.params, empty_roa
        )
        fine, _ = experiment.compute_return_map(
            velocities, actions, self.params, empty_roa, timestep=0.0025
        )
        self.assertTrue(np.all(terminal == -1))
        np.testing.assert_allclose(fine, expected, atol=1e-4, rtol=0)
        self.assertLess(
            np.nanmax(np.abs(fine - expected)), np.nanmax(np.abs(coarse - expected))
        )


class MaximumPolicyTests(unittest.TestCase):
    def make_table(self, successor, terminal):
        successor = np.array(successor, dtype=float)
        n, m = successor.shape
        return experiment.LookupTable(
            np.arange(n, dtype=float),
            np.arange(m),
            successor,
            np.array(terminal),
            np.full(n, np.inf),
            np.full(n, -1),
        )

    def test_longest_successful_path_ignores_dead_end(self):
        table = self.make_table(
            [[np.nan, np.nan], [0, np.nan], [1, np.nan], [2, 4], [4, np.nan]],
            [[0, 0], [-1, 1], [-1, 1], [-1, -1], [-1, -1]],
        )
        maximum = experiment.maximize_steps(table)
        np.testing.assert_array_equal(maximum.steps, [0, 1, 2, 3, -np.inf])
        self.assertEqual(maximum.policy[3], 0)
        with self.assertRaises(ValueError):
            maximum.choose_action(4)

    def test_cycle_with_capture_exit_has_no_finite_maximum(self):
        table = self.make_table([[0, np.nan]], [[-1, 1]])
        with self.assertRaisesRegex(ValueError, "cycle"):
            experiment.maximize_steps(table)


class BalancingPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = model.generate_params()
        cls.roa = experiment.compute_roa(cls.params)

    def test_roa_bounds_and_interior_stability(self):
        self.assertTrue(self.roa.contains(np.zeros(2)))
        self.assertFalse(self.roa.contains(np.array([2.0, 0.0])))
        self.assertFalse(self.roa.contains(np.array([np.nan, 0.0])))
        theta = (self.roa.theta[:-1] + self.roa.theta[1:]) / 2
        velocity = (self.roa.velocity[:-1] + self.roa.velocity[1:]) / 2
        angles, speeds = np.meshgrid(theta, velocity)
        states = np.array([angles.ravel(), speeds.ravel()])
        states = states[:, self.roa.contains(states)]
        final = experiment.test_balance(states, self.params, timestep=0.0025)
        self.assertTrue(np.all(experiment.is_stabilized(final)))

    def test_lookup_rollouts_and_latched_controller(self):
        maximum = np.sqrt(2 * self.params["gravity"] / self.params["length"])
        table = experiment.build_lookup_table(
            np.linspace(0, maximum, 101),
            np.linspace(*experiment.ALPHA_BOUNDS, 17),
            self.params,
            self.roa,
        )
        self.assertEqual(table.steps[0], 0)
        counts = experiment.validate_policy(
            table, np.linspace(0, maximum, 23), self.params, self.roa
        )
        self.assertTrue(np.all(np.isfinite(counts)))
        history = experiment.simulate_policy(3.0, table, self.params, self.roa)
        balancing = history[:, 7].astype(bool)
        self.assertTrue(balancing[-1])
        self.assertTrue(np.all(np.diff(balancing.astype(int)) >= 0))
        self.assertTrue(np.all(history[~balancing, 4] == 0))
        self.assertTrue(np.all(np.diff(history[balancing, 8]) == 0))
        self.assertTrue(experiment.is_stabilized(history[-1, 1:3]))
        self.assertGreaterEqual(history[-1, 8], 3)
        maximum_table = experiment.maximize_steps(table)
        maximum_history = experiment.simulate_policy(
            3.0, maximum_table, self.params, self.roa
        )
        self.assertGreater(maximum_history[-1, 8], history[-1, 8])
        self.assertTrue(experiment.is_stabilized(maximum_history[-1, 1:3]))
        self.assertTrue(np.all(maximum_history[maximum_history[:, 7] == 0, 4] == 0))
        self.assertTrue(
            np.all(
                (history[:, 3] >= experiment.ALPHA_BOUNDS[0])
                & (history[:, 3] <= experiment.ALPHA_BOUNDS[1])
            )
        )


if __name__ == "__main__":
    unittest.main()
