# Synergy Expansion — Numerical Example Replication

Python replication of the numerical example in:

> F. Tessari, A. M. West Jr., N. Hogan, "Explaining human motor coordination via the synergy expansion hypothesis," *PNAS* 122, e2501705122 (2025). https://doi.org/10.1073/pnas.2501705122

The example models a planar 3-link arm (upper arm, forearm, hand) actuated by 10 muscles with constant moment arms. This repository rebuilds it step by step: kinematics, statics, muscle-force distribution, and comparison of solutions.

## Status

| Stage | Content | Status |
|---|---|---|
| 1 | Kinematics: forward kinematics, analytic and numerical Jacobian, plotting | Done |
| 2 | Null space and statics (τ = Jᵀ F) | Planned |
| 3 | Muscle model and force distribution | Planned |
| 4 | Solution comparison (DIEM) | Planned |

## Structure

| File | Content |
|---|---|
| `kinematics.py` | Forward kinematics and 2×3 Jacobian (analytic and numerical) of the 3-link arm |
| `plotting.py` | Arm configuration plot |
| `test_kinematics.py` | Tests: fixed configurations, 2-link limit case (`l3 = 0`), analytic vs numerical Jacobian on random configurations |

## Conventions

- Angles in radians, relative: each joint angle is measured from the previous link.
- Shoulder at the origin. Points: `shoulder`, `elbow`, `wrist`, `end_effector` (S, E, W, EE in the paper).
- Link lengths default to 1.

## Usage

Requires Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_kinematics.py
```