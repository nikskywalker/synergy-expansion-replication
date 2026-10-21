import numpy as np
from scipy.linalg import null_space
from muscles import muscle_force_to_torque, Jm

# Test1: single mono-articular muscle (sf = 100 N, all others 0).
# Expected torques by hand: moment arm x force on the shoulder only,
# 0.04 m x 100 N = [4, 0, 0] Nm.
Fm_sf = np.zeros(10)
Fm_sf[0] = 100 # array in which every element is zero but sf

expected_tau_prod_sf = np.array([4, 0, 0])
tau_prod_sf = muscle_force_to_torque(Fm_sf)

assert np.allclose(tau_prod_sf, expected_tau_prod_sf)


# Test2: single bi-articular muscle (bf = 100 N, all others 0).
# It crosses shoulder and elbow, so it acts on both:
# [0.04 x 100, 0.03 x 100, 0] = [4, 3, 0] Nm.
Fm_bf = np.zeros(10)
Fm_bf[2] = 100 # array in which every element is zero but bf

expected_tau_prod_bf = np.array([4, 3, 0])
tau_prod_bf = muscle_force_to_torque(Fm_bf)

assert np.allclose(tau_prod_bf, expected_tau_prod_bf)


# Test3: co-contraction of a flexor-extensor pair (sf = se = 100 N).
# Equal and opposite torques on the shoulder cancel out: [0, 0, 0] Nm.
Fm_sfse = np.zeros(10)
Fm_sfse[[0,1]] = 100 # array in which every element is zero but sf and se 

expected_tau_prod_sfse = np.array([0, 0, 0])
tau_prod_sfse = muscle_force_to_torque(Fm_sfse)

assert np.allclose(tau_prod_sfse, expected_tau_prod_sfse)


# Test4: null space of Jm^T, i.e. muscle force combinations that produce
# zero joint torque (co-contraction). With 10 muscles and 3 joints
# (Jm of rank 3) its dimension must be 10 - 3 = 7, and every column n
# must give Jm^T @ n = 0.
N = null_space(Jm.T)

null_torques = Jm.T@N
assert N.shape == (10,7) 
assert np.allclose(null_torques, np.zeros((3,7)))

print("Test passed.")


