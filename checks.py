import numpy as np

# these should be equal
omegaNodal = np.load('data/omega_nodal.npy')
dryNaturalFrequenciesSquaredNodal = np.load('data/dryNaturalFrequenciesSquared_nodal.npy')
dryVibrationModesNormalizedNodal = np.load('data/dryVibrationModesNormalized_nodal.npy')

omegaAllModes = np.load('data/omega_allmodes.npy')
dryNaturalFrequenciesSquaredAllModes = np.load('data/dryNaturalFrequenciesSquared_allmodes.npy')
dryVibrationModesNormalizedAllModes = np.load('data/dryVibrationModesNormalized_allmodes.npy')

# these should be a change of basis away from one another
addedMassNodal = np.load('data/added_mass_nodal.npy')
radiationDampingNodal = np.load('data/radiation_damping_nodal.npy')
excitationForceNodal = np.load('data/excitation_force_nodal.npy')

addedMassAllModes = np.load('data/added_mass_allmodes.npy')
radiationDampingAllModes = np.load('data/radiation_damping_allmodes.npy')
excitationForceAllModes = np.load('data/excitation_force_allmodes.npy')

print(np.max(np.abs(dryNaturalFrequenciesSquaredNodal - dryNaturalFrequenciesSquaredAllModes)))
print(np.max(np.abs(dryVibrationModesNormalizedNodal - dryVibrationModesNormalizedAllModes)))

addedMassModalFromNodal = np.zeros_like(addedMassNodal)
radiationDamingModalFromNodal = np.zeros_like(radiationDampingNodal)
excitationForceModalFromNodal = np.zeros_like(excitationForceNodal)
for i in range(omegaAllModes.size):
    addedMassModalFromNodal[i, :, :] = dryVibrationModesNormalizedNodal.transpose() @ addedMassNodal[i, :, :] @ dryVibrationModesNormalizedNodal
    radiationDamingModalFromNodal[i, :, :] = dryVibrationModesNormalizedNodal.transpose() @ radiationDampingNodal[i, :, :] @ dryVibrationModesNormalizedNodal
    excitationForceModalFromNodal[i, :, :] = excitationForceNodal[i, :, :] @ dryVibrationModesNormalizedNodal

print(np.max(np.abs((addedMassModalFromNodal - addedMassAllModes) / addedMassModalFromNodal)))
print(np.max(np.abs(radiationDamingModalFromNodal - radiationDampingAllModes)))
print(np.max(np.abs(excitationForceModalFromNodal - excitationForceAllModes)))

