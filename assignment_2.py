from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
print("STARTING ASSIGNMENT 2")


params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}


def compute_ankle_torque(state, params):
    theta, theta_dot = state

    m = params["mass"]
    g = params["gravity"]
    l = params["length"]

    Kp = 1.0
    Kd = 1.0

    torque = ( -m * g * l * np.sin(theta) - m * l**2 * (Kp * theta + Kd * theta_dot))

    min_torque = -0.1 * m * g * l
    max_torque = 0.05 * m * g * l

    return np.clip(torque, min_torque, max_torque)



def test_balance(initial_state, params, timestep=1e-4, sim_time=3.0):

    state = initial_state.copy()
    test_params = params.copy()

    n_timesteps = round(sim_time / timestep)

    for _ in range(n_timesteps):
        test_params["ankle_torque"] = compute_ankle_torque(state, test_params)

        state = state + timestep * model.dynamics(0, state, test_params)

    return state



def is_stabilized(final_state):
    theta, theta_dot = final_state

    return (abs(theta) < 0.01 and abs(theta_dot) < 0.01)

theta_values = np.linspace(-0.15, 0.15, 25)
velocity_values = np.linspace(-0.5, 0.5, 25)

roa = np.zeros(
    (len(velocity_values), len(theta_values)),
    dtype=bool,
)

for i, theta_dot in enumerate(velocity_values):

    for j, theta in enumerate(theta_values):

        initial_test_state = np.array([
            theta,
            theta_dot,
        ])

        final_state = test_balance(
            initial_test_state,
            params,
        )

        roa[i, j] = is_stabilized(final_state)

plt.figure()

plt.imshow(
    roa,
    origin="lower",
    extent=[
        theta_values[0],
        theta_values[-1],
        velocity_values[0],
        velocity_values[-1],
    ],
    aspect="auto",
)

plt.xlabel("theta (rad)")
plt.ylabel("theta_dot (rad/s)")
plt.title("Region of Attraction")
plt.show()


def state_in_roa(state, theta_values, velocity_values, roa):
    theta, theta_dot = state

    # Outside our tested grid
    if (
        theta < theta_values[0]
        or theta > theta_values[-1]
        or theta_dot < velocity_values[0]
        or theta_dot > velocity_values[-1]
    ):
        return False

    theta_index = np.argmin(np.abs(theta_values - theta))
    velocity_index = np.argmin(np.abs(velocity_values - theta_dot))

    return roa[velocity_index, theta_index]

initial_state = np.array([0.05, 0.0])
timestep = 1e-4
sim_time = 10.0


n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


# Simulation loop. Replace this Euler step with your own integrator as needed.
entered_roa = False

for step, t in enumerate(time_traj[:-1]):

    state = state_traj[:, step]

    # Check if we have entered the Region of Attraction
    if state_in_roa(state, theta_values, velocity_values, roa):
        entered_roa = True

    # Controller OFF outside RoA, ON once inside
    if entered_roa:
        params["ankle_torque"] = compute_ankle_torque(state, params)
    else:
        params["ankle_torque"] = 0.0

    next_state = state + timestep * model.dynamics(t, state, params)

    state_traj[:, step + 1] = next_state

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")

print("Initial state:", state_traj[:, 0])
print("Final state:", state_traj[:, -1])
print("Max theta:", np.max(state_traj[0]))
print("Min theta:", np.min(state_traj[0]))
print("Max velocity:", np.max(np.abs(state_traj[1])))

# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'}.")
plt.show()
