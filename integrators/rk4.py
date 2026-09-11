"""Fourth-order Runge-Kutta integration."""


def step(dynamics, time, state, timestep, params):
    """Advance one timestep with the classical RK4 method."""
    k1 = dynamics(time, state, params)
    k2 = dynamics(time + 0.5 * timestep, state + 0.5 * timestep * k1, params)
    k3 = dynamics(time + 0.5 * timestep, state + 0.5 * timestep * k2, params)
    k4 = dynamics(time + timestep, state + timestep * k3, params)

    return state + timestep / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
