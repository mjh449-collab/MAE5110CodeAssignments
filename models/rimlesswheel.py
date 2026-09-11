import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    spoke_length = params["spoke_length"]

    theta = state[0]
    theta_dot = state[1]

    theta_ddot = gravity / spoke_length * np.sin(theta)

    state_derivative = np.array([theta_dot, theta_ddot])

    return state_derivative


def detect_event(state, params):
    theta = state[0]
    slope_angle = params["slope_angle"]
    number_spokes = params["number_spokes"]

    impact_angle = np.pi / number_spokes + slope_angle

    return theta >= impact_angle


def reset_state(state, params):

    theta_dot = state[1]

    slope_angle = params["slope_angle"]
    number_spokes = params["number_spokes"]

    alpha = np.pi / number_spokes

    theta_after = slope_angle - alpha
    theta_dot_after = theta_dot * np.cos(2 * alpha)

    return np.array([theta_after, theta_dot_after])
