# Project Instructions

## Scope

This repository contains MAE 5110 Python assignments on numerical simulation and legged-robot dynamics. Keep changes focused on the requested assignment and preserve the existing coursework structure.

## Environment and Validation

- Use `uv` for Python commands; do not rely on the system `python` launcher.
- Install the environment with `uv sync --python 3.14`.
- Run scripts with `uv run python <script>`.
- Run tests with `uv run pytest` when tests are present.
- Run Ruff through the project environment when linting is needed: `uv run ruff check .`.

## Codebase Guide

- `models/pendulum.py` contains the pendulum dynamics and energy helpers.
- `models/rimlesswheel.py` contains the rimless-wheel model and should expose clear model functions such as `dynamics(t, state, params)`.
- `assignment_0.py` and `assignment_1.py` are executable assignment entry points.
- Assignment requirements and deliverables are documented in `assignments/assignment_0.md` and `assignments/assignment_1.md`; link to those documents rather than duplicating them.

## Coding Conventions

- Prefer NumPy arrays for state and numerical computations.
- Use descriptive, pronounceable names that communicate purpose.
- Name functions with verbs describing their behavior.
- Define a numerical sanity check before implementing or changing a model.
- Keep simulation, event detection, reset maps, and analysis behavior explicit and easy to inspect.

## Chat Interaction

- Treat inline code suggestions as disabled for this workspace. Do not depend on or recommend VS Code ghost-text suggestions; provide concrete edits, commands, and validation steps instead.
- Make the smallest focused change that satisfies the request, and validate it with the narrowest relevant command.
