import numpy as np

import integrators
from models import pendulum

timestep = 0.01
steps = 100
step = integrators.rk4

initial_state = [0.3, 0]

def simulate(params, state): #simulate for 100 steps
    states = [np.asarray(state, dtype=float)]
    for k in range(steps):
        states.append(step(pendulum.dynamics, k * timestep, states[-1], timestep, params))
    states = np.array(states)

    kinetic, potential = pendulum.calculate_energy(states.T, params)
    return states, kinetic + potential

def set_params(damping, torque): #modify torque/damping as needed
    params = pendulum.generate_params()
    params["damping_coeff"] = damping
    params["torque"] = torque
    return params

def test_energy_conservation(): #energy change should be ~0
    params = set_params(damping=0.0, torque=0.0)
    _, energies = simulate(params, initial_state)

    assert np.isclose(energies[-1] - energies[0], 0.0, atol=1e-6)

def test_damping(): #energy always decreasing if no torque
    params = set_params(damping=0.5, torque=0.0)
    _, energies = simulate(params, initial_state)

    energy_change = np.diff(energies)

    assert np.all(energy_change <= 1e-9)

def test_torque(): #work in = change in energy
    torque = 0.5
    params = set_params(damping=0.0, torque = torque)
    states, energies = simulate(params, initial_state)

    energy_gained = energies[-1] - energies[0]
    work = torque * (states[-1, 0] - states[0, 0])

    assert np.isclose(energy_gained, work, rtol=1e-4, atol=1e-6)