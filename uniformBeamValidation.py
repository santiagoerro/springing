import numpy as np
import numpy.linalg as la
import springing as spr
from scipy.linalg import eigh
import matplotlib.pyplot as plt



# beam properties
beamLength = 1
beamSegments = 40

# section properties
verticalAreaMoments = np.ones([beamSegments])
horizontalAreaMoments = np.ones([beamSegments])
sectionalAreas = np.ones([beamSegments]) * 1e10
verticalShearAreaFractions = np.ones([beamSegments]) * 5/6
horizontalShearAreaFractions = np.ones([beamSegments]) * 5/6

# material properties
youngsModulus = 1
shearModulus = youngsModulus / (2 * (1 + 0.26))

beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(0, beamLength, beamSegments + 1)
beamDefinition['crossSectionAreas'] = sectionalAreas
beamDefinition['verticalAreaMoments'] = verticalAreaMoments
beamDefinition['horizontalAreaMoments'] = horizontalAreaMoments
beamDefinition['verticalTimoshenkoCoefs'] = verticalShearAreaFractions
beamDefinition['horizontalTimoshenkoCoefs'] = horizontalShearAreaFractions
beamDefinition['torsionConstants'] = np.ones([beamSegments])
beamDefinition['warpingConstants'] = np.ones([beamSegments])
beamDefinition['youngsModulus'] = youngsModulus
beamDefinition['shearModulus'] = shearModulus
beamDefinition['zNeutralAxis'] = 0
beamDefinition['zTwistCenter'] = 0
beamDefinition['linearDensities'] = np.ones([beamSegments])



beam = spr.Beam(beamDefinition)

# natural frequencies
dryNaturalFrequenciesSquared, vibrationModesNormalized = eigh(beam.stiffnessMatrix, beam.massMatrix)
dryVerticalBendingNaturalFrequencies = np.sqrt(dryNaturalFrequenciesSquared[6::2])

# internal force distributions
# clamped at initial end
clampedStiffnessMatrix = np.zeros([6 * beamSegments, 6 * beamSegments])
clampedStiffnessMatrix = beam.stiffnessMatrix[6:, 6:]
# 1N point force at final end towards negative z
clampedForcingVector = np.zeros([6 * beamSegments])
clampedForcingVector[6 * (beamSegments - 1) + 2] = -1
clampedNodalDisplacements = la.solve(clampedStiffnessMatrix, clampedForcingVector)
nodalDisplacements = np.zeros([6 * (beamSegments + 1)])
nodalDisplacements[6:] = clampedNodalDisplacements

x = np.linspace(0, beamLength, 500)
verticalBendingMoment = beam.InternalForce(x, nodalDisplacements, 'mv')
verticalShearForce = beam.InternalForce(x, nodalDisplacements, 'sv')



print()
print('Dry vertical bending natural frequencies')
print('Number     Frequency (Hz)')
for i in range(10):
    print('%2d         %.2f'%(i+1, dryVerticalBendingNaturalFrequencies[i]))

plt.figure()
plt.title('Vertical bending moment distribution clamped start, 1N point force end.')
plt.axhline(color = 'k', linewidth = 1)
plt.plot(x, verticalBendingMoment, 'b')
plt.xlabel('x [m]')
plt.ylabel('Vertical bending moment [Nm]')

plt.figure()
plt.title('Vertical shear force distribution clamped start, 1N point force end.')
plt.axhline(color = 'k', linewidth = 1)
plt.plot(x, verticalShearForce, 'b')
plt.xlabel('x [m]')
plt.ylabel('Vertical shear force [N]')

plt.show()