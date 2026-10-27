import numpy as np
from kinematics import forward_kinematics, jacobian, numerical_jacobian, force_to_torque, jacobian_null_space

# Test1: 0°, 90°, 0° and l1=l2=l3=1
elbow1, wrist1, end_effector1 = forward_kinematics(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
jac1 = jacobian(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))

expected_elbow1 = np.array([1., 0.])
expected_wrist1 = np.array([1., 1.]) 
expected_end_effector1 = np.array([1., 2.])

expected_jac1 = np.array([[-2., -2., -1.], [1., 0., 0.]])


assert np.allclose(elbow1, expected_elbow1)
assert np.allclose(wrist1, expected_wrist1)
assert np.allclose(end_effector1, expected_end_effector1)
assert np.allclose(jac1, expected_jac1)


# Test2: 0°, 90°, 45° and l1=l2=1 and l3=0
elbow2, wrist2, end_effector2 = forward_kinematics(np.deg2rad(0), np.deg2rad(90), np.deg2rad(45), l3=0)
jac2 = jacobian(np.deg2rad(0), np.deg2rad(90), np.deg2rad(45), l3=0)

expected_elbow2 = np.array([1., 0.])
expected_wrist2 = np.array([1., 1.]) 
expected_end_effector2 = np.array([1., 1.])

expected_jac2 = np.array([[-1, -1, 0],
                          [1, 0, 0]])

assert np.allclose(elbow2, expected_elbow2)
assert np.allclose(wrist2, expected_wrist2)
assert np.allclose(end_effector2, expected_end_effector2)
assert np.allclose(jac2, expected_jac2)


# Test3: analytic vs numerical Jacobian on 100 random configurations
# (random angles in [-pi, pi], random lengths in [0.5, 2])
rng = np.random.default_rng(21)

for i in range(100):
    q1 = rng.uniform(-np.pi, +np.pi)
    q2 = rng.uniform(-np.pi, +np.pi)
    q3 = rng.uniform(-np.pi, +np.pi)
    l1 = rng.uniform(0.5, 2)
    l2 = rng.uniform(0.5, 2)
    l3 = rng.uniform(0.5, 2)

    an_jac = jacobian(q1, q2, q3, l1, l2, l3)
    num_jac = numerical_jacobian(q1, q2, q3, l1, l2, l3)

    assert np.allclose(an_jac, num_jac)


# Test4: joint torques for F = [10, 0] N at 0°, 90°, 0° (l1 = l2 = l3 = 1),
# compared with values computed by hand
F = np.array([10, 0])

tau = force_to_torque(F, np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
expected_tau = [-20, -20, -10]

assert np.allclose(tau, expected_tau)


# Test5: J^T F vs geometric torques (r_x * F_y - r_y * F_x) on 100 random
# configurations and forces (angles in [-pi, pi], lengths in [0.5, 2], forces in [-10, 10] N)

for i in range(100):
    q1 = rng.uniform(-np.pi, +np.pi)
    q2 = rng.uniform(-np.pi, +np.pi)
    q3 = rng.uniform(-np.pi, +np.pi)
    l1 = rng.uniform(0.5, 2)
    l2 = rng.uniform(0.5, 2)
    l3 = rng.uniform(0.5, 2)
    F = np.array([rng.uniform(-10, +10), rng.uniform(-10, +10)])

    shoulder = np.array([0, 0])
    elbow, wrist, end_effector = forward_kinematics(q1, q2, q3, l1, l2, l3)
    r_x = np.array([end_effector[0] - shoulder[0], end_effector[0] - elbow[0], end_effector[0] - wrist[0]])
    r_y = np.array([end_effector[1] - shoulder[1], end_effector[1] - elbow[1], end_effector[1] - wrist[1]])

    tau_formula = r_x*F[1] - r_y*F[0] 
    tau = force_to_torque(F, q1, q2, q3, l1, l2, l3)

    assert np.allclose(tau, tau_formula)


# Test6: null space of J at 0°, 90°, 0° (l1 = l2 = l3 = 1), compared with the
# vector computed by hand, (0, 1, -2), normalized. scipy returns a unit vector
# with arbitrary sign, so both n and -n are accepted. Also checks J @ n = 0.
n_column = jacobian_null_space(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
n = n_column[:, 0]

expected_n = np.array([0, 1, -2])
expected_n_norm = np.linalg.norm(expected_n)
expected_n = expected_n/expected_n_norm # dividing by the norm to obtain normalized vector to compare with n

J = jacobian(np.deg2rad(0), np.deg2rad(90), np.deg2rad(0))
ee_velocity = J@n # should be 0

assert np.allclose(n, expected_n) or np.allclose(-n, expected_n)
assert np.allclose(ee_velocity, np.zeros(2))


# Test7: null space on 100 random configurations, checked with forward
# kinematics only (no Jacobian). Moving the joints by a small step eps along
# n must leave the end-effector still: displacement ~eps^2, while a wrong n
# would give ~eps. Displacement is compared with zero so that only the
# absolute tolerance of allclose (1e-8) applies.
eps = 1e-6

for i in range(100):
    q1 = rng.uniform(-np.pi, +np.pi)
    q2 = rng.uniform(-np.pi, +np.pi)
    q3 = rng.uniform(-np.pi, +np.pi)
    l1 = rng.uniform(0.5, 2)
    l2 = rng.uniform(0.5, 2)
    l3 = rng.uniform(0.5, 2)
    
    _, _, end_effector = forward_kinematics(q1, q2, q3, l1, l2, l3)

    n_column = jacobian_null_space(q1, q2, q3, l1, l2, l3)
    n = n_column[:,0]
    q_new = np.array([q1 + eps*n[0], q2 + eps*n[1], q3 + eps*n[2]])

    _, _, end_effector_new = forward_kinematics(q_new[0], q_new[1], q_new[2], l1, l2, l3)

    assert np.allclose(end_effector_new - end_effector, np.zeros(2))


print('Test passed.')
