import numpy as np
from muscles import muscle_force_to_torque

# Testing monoarticular muscle
Fm_sf = np.zeros(10)
Fm_sf[0] = 100 # array in which every element is zero but sf

expected_tau_prod_sf = np.array([4, 0, 0])
tau_prod_sf = muscle_force_to_torque(Fm_sf)

assert np.allclose(tau_prod_sf, expected_tau_prod_sf)


# Testing biarticular muscle
Fm_bf = np.zeros(10)
Fm_bf[2] = 100 # array in which every element is zero but bf

expected_tau_prod_bf = np.array([4, 3, 0])
tau_prod_bf = muscle_force_to_torque(Fm_bf)

assert np.allclose(tau_prod_bf, expected_tau_prod_bf)


# Testing a co-contraction situation
Fm_sfse = np.zeros(10)
Fm_sfse[[0,1]] = 100 # array in which every element is zero but sf and se 

expected_tau_prod_sfse = np.array([0, 0, 0])
tau_prod_sfse = muscle_force_to_torque(Fm_sfse)

assert np.allclose(tau_prod_sfse, expected_tau_prod_sfse)


print("Test passed.")


