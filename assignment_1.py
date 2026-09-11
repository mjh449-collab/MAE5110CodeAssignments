import matplotlib.pyplot as plt
import numpy as np

from integrators import rk4 as integrator
from models import rimlesswheel as model


params = {
    "gravity": 9.81,
    "spoke_length": 1.0,
    "slope_angle": 0.1,
    "number_spokes": 8
}


# Convergence

def check_convergence(impact_velocities, tolerance=0.01):
    if len(impact_velocities) < 5:
        return False

    recent_impacts = impact_velocities[-5:]

    return max(recent_impacts) - min(recent_impacts) < tolerance


# Impact refinement

def refine_impact(state_before,time,timestep,params,iterations=12):
    """
    Use bisection to locate an impact within one coarse timestep.
    """

    lower_time = 0.0
    upper_time = timestep

    for _ in range(iterations):

        middle_time = (lower_time + upper_time) / 2

        middle_state = integrator.step(model.dynamics,time,state_before,middle_time,params)

        if model.detect_event(middle_state, params):
            upper_time = middle_time
        else:
            lower_time = middle_time

    impact_time = upper_time

    impact_state = integrator.step(model.dynamics,time,state_before,impact_time,params)

    return impact_state, impact_time

# General simulation

def simulate(initial_state,params,timestep=0.005,sim_time=10,early_stopping=True):
    state = initial_state.copy()
    impact_velocities = []

    time = 0.0

    while time < sim_time:

        timestep_current = min(timestep,sim_time - time)

        # If initial condition is exactly at an impact
        if model.detect_event(state, params):

            state = model.reset_state(state,params)

            impact_velocities.append(state[1])

            if (early_stopping and check_convergence(impact_velocities)):
                break

        state_before = state.copy()

        # Take one normal coarse RK4 step
        state_after = integrator.step(model.dynamics,time,state_before,timestep_current,params)

        # Check whether impact occurred during this step
        impact_crossed = (
            not model.detect_event(state_before, params)
            and model.detect_event(state_after, params)
        )

        if impact_crossed:

            # Find impact accurately
            impact_state, impact_time = refine_impact(state_before,time,timestep_current,params)

            # Apply collision reset
            state = model.reset_state(impact_state,params)

            # Record POST-impact angular velocity
            impact_velocities.append(state[1])

            # Finish unused portion of coarse timestep
            remaining_time = (timestep_current - impact_time)

            if remaining_time > 0:

                state = integrator.step(model.dynamics,time + impact_time,state,remaining_time,params)

        else:
            state = state_after

        time += timestep_current

        if (early_stopping and check_convergence(impact_velocities)):
            break

    return state, impact_velocities


# Region of Attraction
def classify_initial_state(initial_state,params,timestep=0.005,sim_time=10):
    _, impact_velocities = simulate(initial_state,params,timestep=timestep,sim_time=sim_time,early_stopping=True)

    return int(check_convergence(impact_velocities))


def calculate_roa(params,theta_count=40,velocity_count=50,timestep=0.005,sim_time=10):
    alpha = np.pi / params["number_spokes"]
    gamma = params["slope_angle"]

    theta_values = np.linspace(gamma - alpha,gamma + alpha,theta_count)

    theta_dot_values = np.linspace(0,5,velocity_count)

    roa = np.zeros((len(theta_dot_values),len(theta_values)))

    for i, theta_dot in enumerate(theta_dot_values):

        for j, theta in enumerate(theta_values):

            initial_state = np.array([
                theta,
                theta_dot
            ])

            roa[i, j] = classify_initial_state(
                initial_state,
                params,
                timestep=timestep,
                sim_time=sim_time
            )

    return theta_values, theta_dot_values, roa


def plot_roa(params):

    theta_values, theta_dot_values, roa = calculate_roa(params,theta_count=40,velocity_count=50,timestep=0.005,sim_time=10)

    plt.figure()

    plt.imshow(roa,origin="lower",aspect="auto",extent=[theta_values[0],theta_values[-1],theta_dot_values[0],theta_dot_values[-1]])

    plt.xlabel("Initial theta (rad)")
    plt.ylabel("Initial angular velocity (rad/s)")
    plt.title("Region of Attraction")
    plt.colorbar(label="Attractor")



# One-step Poincare map

def one_step_map(theta_dot,params,timestep=0.005,max_time=5):
    alpha = np.pi / params["number_spokes"]
    gamma = params["slope_angle"]

    # Begin immediately after impact
    state = np.array([gamma - alpha,theta_dot])

    time = 0.0

    while time < max_time:

        state_before = state.copy()

        state_after = integrator.step(model.dynamics,time,state_before,timestep,params)

        impact_crossed = (
            not model.detect_event(state_before, params)
            and model.detect_event(state_after, params)
        )

        if impact_crossed:

            impact_state, _ = refine_impact(state_before,time,timestep,params)

            state_after_impact = model.reset_state(impact_state,params)

            return state_after_impact[1]

        state = state_after
        time += timestep

    return np.nan



# Return map and fixed point

def calculate_return_map(params):
    theta_dot_values = np.linspace(0.1,3,100)

    next_theta_dot_values = np.array([one_step_map(theta_dot, params) for theta_dot in theta_dot_values])

    return theta_dot_values, next_theta_dot_values


def find_fixed_point(params):

    theta_dot_values, next_theta_dot_values = (calculate_return_map(params))

    difference = np.abs(next_theta_dot_values - theta_dot_values)

    fixed_point_index = np.nanargmin(difference)

    return theta_dot_values[fixed_point_index]


def plot_return_map(params):

    theta_dot_values, next_theta_dot_values = (calculate_return_map(params))

    difference = np.abs(next_theta_dot_values - theta_dot_values)

    fixed_point_index = np.nanargmin(difference)

    fixed_point = theta_dot_values[fixed_point_index]

    fixed_point_next = next_theta_dot_values[fixed_point_index]

    plt.figure()

    plt.plot(theta_dot_values,next_theta_dot_values,label="Return map")

    plt.plot(theta_dot_values,theta_dot_values,"--",label="Identity line")

    plt.scatter(fixed_point,fixed_point_next,label="Fixed point")

    plt.xlabel("Current angular velocity (rad/s)")

    plt.ylabel("Next angular velocity (rad/s)")

    plt.title("Step-to-Step Return Map")

    plt.legend()

    print("Fixed point:",fixed_point)

    return fixed_point



# Floquet multiplier
def calculate_floquet(fixed_point,params,epsilon=0.001):
    velocity_below = (fixed_point - epsilon)

    velocity_above = (fixed_point + epsilon)

    next_below = one_step_map(velocity_below,params)

    next_above = one_step_map(velocity_above,params)

    floquet_multiplier = (next_above - next_below) / (2 * epsilon)

    return floquet_multiplier



# Slope sweep
def sweep_slope(params):

    gamma_values = np.deg2rad(np.linspace(1, 10, 10))

    roa_sizes = []
    floquet_values = []

    for i, gamma in enumerate(gamma_values):

        print(
            f"Slope {i + 1}/{len(gamma_values)}"
        )

        test_params = params.copy()

        test_params["slope_angle"] = gamma

        # Coarse grid for faster parameter sweep
        _, _, roa = calculate_roa(test_params,theta_count=15,velocity_count=20,timestep=0.005,sim_time=6)

        roa_sizes.append(np.mean(roa == 1))

        fixed_point = find_fixed_point(test_params)

        floquet = calculate_floquet(fixed_point,test_params)

        floquet_values.append(floquet)

    plt.figure()

    plt.plot(np.rad2deg(gamma_values),roa_sizes,"o-")

    plt.xlabel("Slope (degrees)")
    plt.ylabel("Fraction of state space attracted to rolling")

    plt.title("Effect of Slope on Region of Attraction")

    plt.figure()

    plt.plot(np.rad2deg(gamma_values),floquet_values,"o-")

    plt.xlabel("Slope (degrees)")
    plt.ylabel("Floquet multiplier")

    plt.title("Effect of Slope on Local Convergence")


# Number of spokes sweep


def sweep_spokes(params):

    spoke_values = range(6, 13)

    roa_sizes = []
    floquet_values = []

    for i, number_spokes in enumerate(
        spoke_values
    ):

        print(
            f"Spokes {i + 1}/7"
        )

        test_params = params.copy()

        test_params["number_spokes"] = number_spokes

        # Coarse grid for faster parameter sweep
        _, _, roa = calculate_roa(test_params,theta_count=15,velocity_count=20,timestep=0.005,sim_time=6)

        roa_sizes.append(np.mean(roa == 1))

        fixed_point = find_fixed_point(test_params)

        floquet = calculate_floquet(fixed_point,test_params)

        floquet_values.append(floquet)

    plt.figure()

    plt.plot(list(spoke_values),roa_sizes,"o-")

    plt.xlabel("Number of spokes")
    plt.ylabel("Fraction of state space attracted to rolling")

    plt.title("Effect of Number of Spokes on Region of Attraction")

    plt.figure()

    plt.plot(list(spoke_values),floquet_values,"o-")

    plt.xlabel("Number of spokes")
    plt.ylabel("Floquet multiplier")

    plt.title(
        "Effect of Number of Spokes on Local Convergence"
    )



# Run

if __name__ == "__main__":

    print("Calculating Region of Attraction...")
    plot_roa(params)

    print("Calculating return map...")
    fixed_point = plot_return_map(params)

    floquet = calculate_floquet(fixed_point,params)

    print("Floquet multiplier:",floquet)

    print("Running slope sweep...")
    sweep_slope(params)

    print("Running spoke sweep...")
    sweep_spokes(params)

    plt.show()


#sanity check, theta over time

'''import numpy as np
from models import rimlesswheel as model
from integrators import rk4 as integrator

params = {
    "gravity": 9.81,
    "spoke_length": 1.0,
    "slope_angle": 0.1,
    "number_spokes": 8
}

timestep = 0.001
sim_time = 5

time = np.arange(0, sim_time, timestep)
state = np.array([0.0, 0.5])

state_traj = np.zeros((2, len(time)))
state_traj[:, 0] = state

for i in range(len(time) - 1):

    state = integrator.step(
        model.dynamics,
        time[i],
        state,
        timestep,
        params
    )

    if model.detect_event(state, params):
        state = model.reset_state(state, params)

    state_traj[:, i + 1] = state

import matplotlib.pyplot as plt

#impact angle sanity check
impact_angle = params["slope_angle"] + np.pi / params["number_spokes"]

state = np.array([impact_angle, 1.0])
    
assert model.detect_event(state, params)
print("Impact event detected correctly")

plt.figure()
plt.plot(time, state_traj[0])
plt.xlabel("Time (s)")
plt.ylabel("Theta (rad)")
plt.show()

plt.figure()
plt.plot(time, state_traj[1])
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.show()'''