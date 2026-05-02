import numpy as np
import springing as spr
from scipy.linalg import eigh



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

dryNaturalFrequenciesSquared, vibrationModesNormalized = eigh(beam.stiffnessMatrix, beam.massMatrix)
dryVerticalBendingNaturalFrequencies = np.sqrt(dryNaturalFrequenciesSquared[6::2])

print()
print('Dry vertical bending natural frequencies')
print('Number     Frequency (Hz)')
for i in range(10):
    print('%2d         %.2f'%(i+1, dryVerticalBendingNaturalFrequencies[i]))