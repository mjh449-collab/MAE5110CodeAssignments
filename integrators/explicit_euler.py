"""Explicit Euler integration."""


def step(dynamics, time, state, timestep, params):
    """Advance one timestep with the explicit Euler method."""
    return state + timestep * dynamics(time, state, params)
