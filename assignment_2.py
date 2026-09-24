"""Hybrid walker balancing with minimum- and maximum-footstep policies.

Run: uv run python assignment_2.py [--no-show] [--skip-animation]
Artifacts go to output/assignment_2. The return section is theta=0, independent
of alpha and transverse because theta'=velocity>0. Zero velocity at upright
is a terminal equilibrium only.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

ALPHA_BOUNDS = (np.pi / 8, np.pi / 7)
TIMESTEP = 0.005


def compute_ankle_torque(state, params):
    """Cancel gravity and impose theta'' + 2 theta' + theta = 0, if unsaturated."""
    theta, velocity = state
    m, g, length = (params[key] for key in ("mass", "gravity", "length"))
    torque = -m * g * length * np.sin(theta) - m * length**2 * (theta + 2 * velocity)
    return np.clip(torque, -0.1 * m * g * length, 0.05 * m * g * length)


def integrate(state, params, dt, balance=False):
    """RK4 for one state or a batch; recompute feedback at every RK stage."""

    def derivative(value):
        torque = compute_ankle_torque(value, params) if balance else 0.0
        return model.dynamics(0, value, {**params, "ankle_torque": torque})

    k1 = derivative(state)
    k2 = derivative(state + dt * k1 / 2)
    k3 = derivative(state + dt * k2 / 2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def test_balance(initial_state, params, timestep=TIMESTEP, sim_time=12.0):
    """Return the controlled endpoint; no footstrikes while balancing."""
    state = np.array(initial_state, dtype=float, copy=True)
    for _ in range(round(sim_time / timestep)):
        state = integrate(state, params, timestep, balance=True)
    return state


def is_stabilized(state):
    return np.all(np.abs(state) < 0.01, axis=0)


@dataclass
class RegionOfAttraction:
    theta: np.ndarray
    velocity: np.ndarray
    stable: np.ndarray

    def contains(self, state):
        """Require all four surrounding vertices to converge; reject out-of-grid states.

        This is a conservative numerical estimate, not a stability certificate.
        """
        theta, velocity = np.asarray(state)
        inside = (
            np.isfinite(theta)
            & np.isfinite(velocity)
            & (theta >= self.theta[0])
            & (theta <= self.theta[-1])
            & (velocity >= self.velocity[0])
            & (velocity <= self.velocity[-1])
        )
        j = np.clip(np.searchsorted(self.theta, theta) - 1, 0, len(self.theta) - 2)
        i = np.clip(
            np.searchsorted(self.velocity, velocity) - 1, 0, len(self.velocity) - 2
        )
        return (
            inside
            & self.stable[i, j]
            & self.stable[i + 1, j]
            & self.stable[i, j + 1]
            & self.stable[i + 1, j + 1]
        )


def compute_roa(params, timestep=TIMESTEP):
    """Sweep in parallel; reject falls and require a full final second settled."""
    # State-space search around upright; these are not alpha or torque limits.
    theta = np.linspace(-0.5, 0.5, 101)  # rad
    velocity = np.linspace(-2.0, 2.0, 161)  # rad/s
    angles, speeds = np.meshgrid(theta, velocity)
    state = np.array([angles.ravel(), speeds.ravel()])
    stable = np.ones(state.shape[1], dtype=bool)
    for step in range(round(12.0 / timestep)):
        state = integrate(state, params, timestep, balance=True)
        stable &= np.abs(state[0]) < np.pi / 2
        if (step + 1) * timestep >= 11.0:
            stable &= is_stabilized(state)
    return RegionOfAttraction(theta, velocity, stable.reshape(angles.shape))


def advance_walker(state, alpha, params, timestep):
    """Advance passive stance with an impact resolved within the timestep.

    Interpolate the crossing time, integrate to it, reset on the exact guard,
    then integrate the remaining time. Takes a batch and one angle per state.
    """
    next_state = integrate(state, params, timestep)
    touchdown = params["incline"] + alpha
    hit = (state[0] < touchdown) & (next_state[0] >= touchdown) & (next_state[1] > 0)
    if np.any(hit):
        fraction = (touchdown[hit] - state[0, hit]) / (
            next_state[0, hit] - state[0, hit]
        )
        impact = integrate(state[:, hit], params, timestep * fraction)
        impact[0] = touchdown[hit]
        reset = model.event_dynamics(impact, {**params, "angle_of_attack": alpha[hit]})
        next_state[:, hit] = integrate(reset, params, timestep * (1 - fraction))
    return next_state, hit


def compute_return_map(
    velocities, actions, params, roa, timestep=TIMESTEP, paired=False
):
    """Simulate to capture, the next forward section crossing, or failure.

    A return contains exactly one footstrike. Capture can precede impact (zero
    steps) or follow it (one step). NaN successor and terminal=-1 mean failure
    or timeout. Paired inputs simulate individual (state, action) pairs.
    """
    if paired:
        speeds, angles = np.broadcast_arrays(velocities, actions)
    else:
        speeds, angles = np.meshgrid(velocities, actions, indexing="ij")
    state = np.array([np.zeros(speeds.size), speeds.ravel()])
    alpha = angles.ravel()
    active = np.ones(speeds.size, dtype=bool)
    impacts = np.zeros(speeds.size, dtype=int)
    successor = np.full(speeds.size, np.nan)
    terminal = np.full(speeds.size, -1, dtype=int)
    for _ in range(round(12.0 / timestep)):
        captured = active & roa.contains(state)
        terminal[captured] = impacts[captured]
        active[captured] = False
        active &= (state[1] > 0) & (np.abs(state[0]) < np.pi / 2)
        indices = np.flatnonzero(active)
        if not indices.size:
            break
        previous = state[:, indices]
        following, hit = advance_walker(previous, alpha[indices], params, timestep)
        impacts[indices] += hit
        state[:, indices] = following
        captured = roa.contains(following)
        terminal[indices[captured]] = impacts[indices[captured]]
        active[indices[captured]] = False
        crossed = (
            (previous[0] < 0) & (following[0] >= 0) & (following[1] > 0) & ~captured
        )
        if np.any(crossed):
            fraction = -previous[0, crossed] / (
                following[0, crossed] - previous[0, crossed]
            )
            section = integrate(previous[:, crossed], params, timestep * fraction)
            successor[indices[crossed]] = section[1]
            active[indices[crossed]] = False
    return successor.reshape(speeds.shape), terminal.reshape(speeds.shape)


def nearest_indices(grid, values):
    upper = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    return upper - (np.abs(values - grid[upper - 1]) <= np.abs(values - grid[upper]))


@dataclass
class LookupTable:
    velocities: np.ndarray
    actions: np.ndarray
    successor: np.ndarray
    terminal: np.ndarray
    steps: np.ndarray
    policy: np.ndarray

    def choose_action(self, velocity):
        if not self.velocities[0] <= velocity <= self.velocities[-1]:
            raise ValueError("Velocity outside the lookup table.")
        index = nearest_indices(self.velocities, velocity)
        if not np.isfinite(self.steps[index]):
            raise ValueError("No stabilizing action found for this velocity.")
        return self.actions[self.policy[index]]


def build_lookup_table(velocities, actions, params, roa, timestep=TIMESTEP):
    successor, terminal = compute_return_map(velocities, actions, params, roa, timestep)
    valid = (
        np.isfinite(successor)
        & (successor >= velocities[0])
        & (successor <= velocities[-1])
    )
    indices = nearest_indices(velocities, np.nan_to_num(successor))
    steps = np.full(len(velocities), np.inf)
    policy = np.full(len(velocities), -1, dtype=int)
    # Synchronous shortest-path propagation: capture, then two steps, etc.
    for _ in range(len(velocities) + 1):
        costs = np.where(terminal >= 0, terminal, np.inf)
        costs = np.minimum(costs, np.where(valid, 1 + steps[indices], np.inf))
        updated = np.min(costs, axis=1)
        improved = updated < steps
        if not np.any(improved):
            break
        steps = updated
    # Resolve equal-cost actions away from capture boundaries. Otherwise the
    # first angle in a row can repeatedly round into a lower-cost cell without
    # achieving that cost in the continuous dynamics.
    for i in np.flatnonzero(np.isfinite(steps)):
        best = np.flatnonzero(costs[i] == steps[i])
        captures = best[terminal[i, best] >= 0]
        policy[i] = (
            captures[len(captures) // 2]
            if captures.size
            else best[np.argmin(successor[i, best])]
        )
    return LookupTable(velocities, actions, successor, terminal, steps, policy)


def maximize_steps(table):
    """Longest successful path to mandatory RoA capture on the sampled graph.

    Failure has value -inf, not a reward for walking forever. A positive cycle
    with an exit to capture makes successful path lengths unbounded on this
    grid; report it rather than treating an iteration limit as a maximum.
    Such a cycle can be a quantization artifact and needs physical verification.
    """
    successor, terminal = table.successor, table.terminal
    valid = (
        np.isfinite(successor)
        & (successor >= table.velocities[0])
        & (successor <= table.velocities[-1])
    )
    indices = nearest_indices(table.velocities, np.nan_to_num(successor))
    steps = np.full(len(table.velocities), -np.inf)
    for _ in range(len(steps) + 1):
        costs = np.maximum(
            np.where(terminal >= 0, terminal, -np.inf),
            np.where(valid, 1 + steps[indices], -np.inf),
        )
        updated = np.max(costs, axis=1)
        if np.array_equal(updated, steps):
            break
        steps = updated
    else:
        raise ValueError(
            "A cycle with an exit to the RoA gives no finite maximum on this "
            "grid. Refine and verify the cycle in the continuous dynamics."
        )
    policy = np.full(len(steps), -1, dtype=int)
    for i in np.flatnonzero(np.isfinite(steps)):
        best = np.flatnonzero(costs[i] == steps[i])
        returns = best[valid[i, best]]
        policy[i] = (
            returns[np.argmax(successor[i, returns])]
            if returns.size
            else best[len(best) // 2]
        )
    return LookupTable(
        table.velocities, table.actions, successor, terminal, steps, policy
    )


def validate_policy(table, probes, params, roa, timestep=TIMESTEP / 2):
    """Execute the policy on unrounded states, including off-grid probes."""
    values = np.array(probes, copy=True)
    counts = np.full(len(values), np.inf)
    active = np.ones(len(values), dtype=bool)
    for step in range(100):
        captured = active & roa.contains(np.array([np.zeros(len(values)), values]))
        counts[captured] = step
        active[captured] = False
        nearest = nearest_indices(table.velocities, values)
        active &= (
            (values >= table.velocities[0])
            & (values <= table.velocities[-1])
            & np.isfinite(table.steps[nearest])
        )
        indices = np.flatnonzero(active)
        if not indices.size:
            break
        actions = table.actions[table.policy[nearest[indices]]]
        successor, terminal = compute_return_map(
            values[indices], actions, params, roa, timestep, paired=True
        )
        captured = terminal >= 0
        counts[indices[captured]] = step + terminal[captured]
        active[indices[captured | ~np.isfinite(successor)]] = False
        returning = ~captured & np.isfinite(successor)
        values[indices[returning]] = successor[returning]
    return counts


def select_grid(params, roa, output):
    """Pick the coarsest tested grid passing independent rollout checks.

    Compare with an 801x65 reference on 61 fixed off-grid speeds: require identical
    reachability, >=95% identical predicted AND executed step counts, <=1 step
    error, and successful rollouts for every probe labeled reachable. This validates the probe set,
    not every continuous initial condition.
    """
    maximum = np.sqrt(2 * params["gravity"] / params["length"])
    probes = np.linspace(0, maximum, 63)[1:-1]
    resolutions = [
        (9, 3),
        (13, 3),
        (25, 5),
        (51, 9),
        (101, 17),
        (201, 33),
        (401, 33),
        (801, 65),
    ]
    tables = []
    for states, controls in resolutions:
        print(
            f"Building return map: {states} velocities x {controls} angles", flush=True
        )
        tables.append(
            build_lookup_table(
                np.linspace(0, maximum, states),
                np.linspace(*ALPHA_BOUNDS, controls),
                params,
                roa,
            )
        )
    reference = tables[-1].steps[nearest_indices(tables[-1].velocities, probes)]
    rows = []
    selected = None
    for table in tables:
        predicted = table.steps[nearest_indices(table.velocities, probes)]
        actual = validate_policy(table, probes, params, roa)
        reachable = np.isfinite(reference)
        agreement = np.mean(np.isfinite(actual) == reachable)
        exact = (
            np.mean(actual[reachable] == reference[reachable])
            if np.any(reachable)
            else 1.0
        )
        error = (
            np.max(np.abs(actual[reachable] - reference[reachable]))
            if np.any(reachable)
            else 0.0
        )
        safe = np.all(np.isfinite(actual[np.isfinite(predicted)]))
        predicted_exact = np.mean(predicted == reference)
        passed = (
            agreement == 1
            and exact >= 0.95
            and predicted_exact >= 0.95
            and error <= 1
            and safe
        )
        row = (
            len(table.velocities),
            len(table.actions),
            agreement,
            exact,
            error,
            int(safe),
            int(passed),
            predicted_exact,
        )
        rows.append(row)
        print(
            f"Grid {row[0]}x{row[1]}: reach agreement={agreement:.3f}, exact executed={exact:.3f}, exact predicted={predicted_exact:.3f}, max error={error}, rollout safe={safe}, pass={passed}",
            flush=True,
        )
        if passed and selected is None:
            selected = table
    np.savetxt(
        output / "grid_validation.csv",
        rows,
        delimiter=",",
        header="states,actions,reachability_agreement,exact_step_fraction,max_step_error,all_predicted_reachable_succeed,passed,exact_prediction_fraction",
        comments="",
        fmt=["%d", "%d", "%.4f", "%.4f", "%.4f", "%d", "%d", "%.4f"],
    )
    if selected is None:
        raise RuntimeError("No grid passed validation; refine before using the policy.")
    return selected


def simulate_policy(
    initial_velocity, table, params, roa, timestep=TIMESTEP, max_time=60.0
):
    """Run walking then latched balancing; retain controls and stance positions."""
    state = np.array([0.0, initial_velocity])
    alpha = table.choose_action(initial_velocity)
    foot = np.zeros(2)
    balancing = False
    capture_time = None
    impacts = 0
    history = []
    for step in range(round(max_time / timestep) + 1):
        time = step * timestep
        if not balancing and roa.contains(state):
            balancing, capture_time = True, time
        torque = compute_ankle_torque(state, params) if balancing else 0.0
        history.append([time, *state, alpha, torque, *foot, balancing, impacts])
        if balancing and time - capture_time >= 12.0:
            if not is_stabilized(state):
                raise RuntimeError("RoA capture did not stabilize the walker.")
            return np.array(history)
        if balancing:
            state = integrate(state, params, timestep, balance=True)
            continue
        following, hit = advance_walker(
            state[:, None], np.array([alpha]), params, timestep
        )
        following = following[:, 0]
        if hit[0]:
            impacts += 1
            foot += (
                2
                * params["length"]
                * np.sin(alpha)
                * np.array([np.cos(params["incline"]), -np.sin(params["incline"])])
            )
        if (
            state[0] < 0 <= following[0]
            and following[1] > 0
            and not roa.contains(following)
        ):
            fraction = -state[0] / (following[0] - state[0])
            section = integrate(state, params, timestep * fraction)
            alpha = table.choose_action(section[1])
        state = following
        if (abs(state[0]) >= np.pi / 2 or state[1] <= 0) and not roa.contains(state):
            raise RuntimeError("Walker failed before reaching the RoA.")
    raise RuntimeError("Simulation timed out before stabilization.")


def compare_step_policies(initial_velocity, minimum, table, params, roa, output):
    """Check the maximum against a finer grid and half-timestep physical rollout.

    The result is a validated numerical maximum for the sampled state/actions,
    not an exact optimality certificate over every continuous landing angle.
    """
    maximum = maximize_steps(table)
    limit = np.sqrt(2 * params["gravity"] / params["length"])
    reference = maximize_steps(
        build_lookup_table(
            np.linspace(0, limit, 801), np.linspace(*ALPHA_BOUNDS, 65), params, roa
        )
    )
    reference_count = reference.steps[
        nearest_indices(reference.velocities, initial_velocity)
    ]
    rows = []
    selected = None
    for candidate in (maximum, reference):
        predicted = candidate.steps[
            nearest_indices(candidate.velocities, initial_velocity)
        ]
        counts = [
            validate_policy(candidate, [initial_velocity], params, roa, dt)[0]
            for dt in (TIMESTEP, TIMESTEP / 2)
        ]
        passed = (
            np.isfinite(predicted)
            and predicted == reference_count
            and all(count == predicted for count in counts)
        )
        rows.append(
            [
                len(candidate.velocities),
                len(candidate.actions),
                predicted,
                *counts,
                reference_count,
                int(passed),
            ]
        )
        if passed and selected is None:
            selected = candidate
    np.savetxt(
        output / "maximum_steps_validation.csv",
        rows,
        delimiter=",",
        header="states,actions,predicted_steps,rollout_steps,half_timestep_steps,reference_steps,passed",
        comments="",
        fmt="%.6g",
    )
    if selected is None:
        raise RuntimeError(
            "Maximum-step policy failed refinement/rollout checks; refine the grid."
        )
    history = simulate_policy(initial_velocity, selected, params, roa)
    if history[-1, 8] != reference_count:
        raise RuntimeError(
            "Full maximum-step trajectory disagrees with the validated return map."
        )
    np.savetxt(
        output / "trajectory_maximum.csv",
        history,
        delimiter=",",
        header="time,theta,velocity,alpha,torque,foot_x,foot_y,balancing,footstrikes",
        comments="",
    )
    np.savez(
        output / "maximum_lookup_table.npz",
        velocities=selected.velocities,
        actions=selected.actions,
        successor=selected.successor,
        terminal=selected.terminal,
        steps=selected.steps,
        policy=selected.policy,
    )
    save_step_comparison(minimum, history, output)
    print(
        f"Same initial velocity: minimum policy {int(minimum[-1, 8])} footstrikes; "
        f"maximum policy {int(history[-1, 8])} footstrikes (refinement and half-timestep checked).",
        flush=True,
    )
    return history


def save_step_comparison(minimum, maximum, output):
    """Compare physical trajectories and mandatory capture for the same start."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for history, color, name in (
        (minimum, "tab:blue", "Minimum"),
        (maximum, "tab:orange", "Maximum"),
    ):
        capture = history[np.flatnonzero(history[:, 7])[0], 0]
        label = f"{name}: {int(history[-1, 8])} footstrikes"
        for ax, column in zip(axes, (1, 2, 8)):
            ax.plot(
                history[:, 0],
                history[:, column],
                color=color,
                label=label,
                drawstyle="steps-post" if column == 8 else "default",
            )
            ax.axvline(capture, color=color, linestyle=":", alpha=0.8)
    for ax, ylabel in zip(axes, ("theta (rad)", "velocity (rad/s)", "footstrikes")):
        ax.set_ylabel(ylabel)
    axes[0].legend()
    axes[0].set_title(
        f"Same initial state: theta = 0, velocity = {minimum[0, 2]:.4f} rad/s\nDotted lines mark RoA entry"
    )
    axes[-1].set_xlabel("time (s)")
    fig.savefig(output / "minimum_maximum_trajectories.png", dpi=180)


def save_poincare_plot(history, output):
    """Mark forward section crossings, excluding impacts and balancing motion."""
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    phase = history[:, 1:3].copy()
    impacts = np.flatnonzero(np.diff(history[:, 8]) > 0) + 1
    phase[impacts] = np.nan
    balancing = history[:, 7].astype(bool)
    for mask, color, label in (
        (~balancing, "tab:blue", "Passive walking"),
        (balancing, "tab:orange", "Ankle balancing"),
    ):
        segment = phase.copy()
        segment[~mask] = np.nan
        ax.plot(segment[:, 0], segment[:, 1], color=color, label=label)
    for index, impact in enumerate(impacts):
        ax.plot(
            history[impact - 1 : impact + 1, 1],
            history[impact - 1 : impact + 1, 2],
            "--",
            color="0.55",
            linewidth=1,
            label="Footstrike reset" if index == 0 else None,
        )

    ax.plot(
        [0, 0],
        [0, 1.08 * np.max(history[:, 2])],
        ":",
        color="tab:green",
        linewidth=2,
        label=r"Section: $\theta=0,\ \dot\theta>0$",
    )
    # Interpolate only continuous, forward crossings during passive walking.
    crossings = np.flatnonzero(
        (history[:-1, 1] < 0)
        & (history[1:, 1] >= 0)
        & (history[1:, 2] > 0)
        & ~balancing[:-1]
        & ~balancing[1:]
        & (np.diff(history[:, 8]) == 0)
    )
    speeds = []
    if history[0, 1] == 0 and history[0, 2] > 0 and not balancing[0]:
        speeds.append(history[0, 2])
    for index in crossings:
        before, after = history[index : index + 2, 1:3]
        fraction = -before[0] / (after[0] - before[0])
        speeds.append(before[1] + fraction * (after[1] - before[1]))
    for index, speed in enumerate(speeds):
        ax.plot(0, speed, "o", color="tab:green", markersize=6)
        ax.annotate(
            rf"$\dot\theta_{{{index}}}={speed:.2f}$ rad/s",
            (0, speed),
            xytext=(15, 12),
            textcoords="offset points",
            fontsize=10,
            color="darkgreen",
        )
        ax.annotate(
            "",
            xy=(0.025, speed),
            xytext=(-0.025, speed),
            arrowprops={"arrowstyle": "->", "color": "tab:green"},
        )
    if np.any(balancing):
        capture = history[np.flatnonzero(balancing)[0], 1:3]
        ax.plot(*capture, "D", color="tab:orange", label="RoA entry")
    ax.plot(0, 0, "ko", markersize=4, label="Standing equilibrium (not a crossing)")
    ax.set(
        xlabel=r"$\theta$ (rad)",
        ylabel=r"$\dot\theta$ (rad/s)",
        title="Poincare section and forward crossings",
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output / "poincare_section.png", dpi=180)


def save_plots(roa, table, history, output):
    save_poincare_plot(history, output)
    fig, ax = plt.subplots(layout="constrained")
    ax.pcolormesh(roa.theta, roa.velocity, roa.stable, shading="nearest", cmap="Greens")
    phase = history[:, 1:3].copy()
    impacts = np.flatnonzero(np.diff(history[:, 8]) > 0) + 1
    phase[impacts] = np.nan  # An impact is a reset, not a continuous orbit.
    ax.plot(
        phase[:, 0],
        phase[:, 1],
        color="tab:blue",
        linewidth=1,
        label="Policy trajectory",
    )
    for index, impact in enumerate(impacts):
        ax.plot(
            history[impact - 1 : impact + 1, 1],
            history[impact - 1 : impact + 1, 2],
            "--",
            color="0.5",
            linewidth=0.8,
            label="Footstrike reset" if index == 0 else None,
        )
    ax.scatter([], [], marker="s", color="darkgreen", label="Converged RoA grid")
    ax.set(
        xlabel="theta (rad)",
        ylabel="angular velocity (rad/s)",
        title="Numerical region of attraction",
    )
    ax.legend()
    fig.savefig(output / "roa.png", dpi=160)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    image = axes[0].pcolormesh(
        table.actions,
        table.velocities,
        np.ma.masked_invalid(table.successor),
        shading="nearest",
    )
    axes[0].set(
        xlabel="angle of attack (rad)",
        ylabel="section velocity (rad/s)",
        title="Return map (capture / failure masked)",
    )
    fig.colorbar(image, ax=axes[0], label="next section velocity (rad/s)")
    axes[1].step(table.velocities, table.steps, where="mid")
    axes[1].set(
        xlabel="section velocity (rad/s)",
        ylabel="minimum footstrikes",
        title="Steps to the balancing RoA",
    )
    fig.savefig(output / "lookup_table.png", dpi=160)
    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained")
    for ax, column, label in zip(
        axes, [1, 2, 4], ["theta (rad)", "velocity (rad/s)", "ankle torque (N m)"]
    ):
        ax.plot(history[:, 0], history[:, column])
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time (s)")
    fig.savefig(output / "trajectory.png", dpi=160)


def save_animation(history, params, output):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    indices = np.searchsorted(history[:, 0], np.arange(0, history[-1, 0], 1 / 25))

    def draw(index):
        time, theta, velocity, alpha, torque, x, y, balancing, _ = history[index]
        model.visualize(
            [theta, velocity],
            {**params, "angle_of_attack": alpha, "ankle_torque": torque},
            ax=ax,
            stance_position=(x, y),
            show_swing=not balancing,
        )
        ax.set_title(f"t = {time:.2f} s")

    animation = FuncAnimation(fig, draw, frames=indices, interval=40, repeat=False)
    animation.save(output / "walker.gif", writer=PillowWriter(fps=25))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--skip-animation", action="store_true")
    args = parser.parse_args()
    params = model.generate_params()
    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)
    print("Computing the balancing RoA...", flush=True)
    roa = compute_roa(params)
    table = select_grid(params, roa, output)
    candidates = np.flatnonzero(np.isfinite(table.steps) & (table.steps >= 3))
    if not candidates.size:
        raise RuntimeError(
            "No initial condition requiring at least three steps was found."
        )
    initial_velocity = table.velocities[candidates[len(candidates) // 2]]
    history = simulate_policy(initial_velocity, table, params, roa)
    compare_step_policies(initial_velocity, history, table, params, roa, output)
    np.savez(
        output / "lookup_table.npz",
        velocities=table.velocities,
        actions=table.actions,
        successor=table.successor,
        terminal=table.terminal,
        steps=table.steps,
        policy=table.policy,
        theta=roa.theta,
        roa_velocity=roa.velocity,
        roa=roa.stable,
    )
    np.savetxt(
        output / "trajectory.csv",
        history,
        delimiter=",",
        header="time,theta,velocity,alpha,torque,foot_x,foot_y,balancing,footstrikes",
        comments="",
    )
    save_plots(roa, table, history, output)
    print(f"Selected grid: {len(table.velocities)} x {len(table.actions)}")
    print(f"Initial section velocity: {initial_velocity:.6f} rad/s")
    print(f"Entered RoA: {bool(history[-1, 7])}; footstrikes: {int(history[-1, 8])}")
    print(f"Final state: {history[-1, 1:3]}")
    if not args.skip_animation:
        save_animation(history, params, output)
    print(f"Saved results to {output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
