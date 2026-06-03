import numpy as np
import capytaine as cpt
import sys
import springing as spr

cpt.set_logging('INFO')



# INPUT

# water properties
waterDensity = 1000

# gravity
gravity = 9.81

# hull geometry
hullLength = 1.52
hullBreadth = 0.22
hullDepth = 0.13
hullDisplacement = 15.6

# mass distribution
pointMasses = np.array([0.5, 0.5, 0.5, 1, 2, 2.77, 1, 1])
pointMassStations = np.array([1, 2.5, 4.5, 6.5, 8, 12, 15.5, 17.5]) # ship divided into 20 stations

# beam properties
beamSegments = 40

# section properties
verticalAreaMoments = np.ones([beamSegments]) * 0.59969e-05
horizontalAreaMoments = np.ones([beamSegments]) * 0.24325e-04
sectionalAreas = np.ones([beamSegments]) * 0.36945e-02
verticalShearAreaFractions = np.ones([beamSegments]) * 0.32148
horizontalShearAreaFractions = np.ones([beamSegments]) * 0.28385
torsionConstants = np.ones([beamSegments]) * 1.5223e-06
warpingConstants = np.ones([beamSegments]) * 4.0778e-08
zCentroidOverBottom = 0.45434e-01
zShearCenterOverBottom = -0.26763e-01

# material properties
youngsModulus = 0.215e10
shearModulus = youngsModulus / (2 * (1 + 0.22))

# wave characteristics
waveHeight = 0.05

waterDepths = np.array([1.8])
omegas = np.array([1, 2, 3, 4, 4.50, 5, 5.59, 5.81, 6.07, 6.37, 6.71, 7.12, 7.4, 7.8, 8.22, 8.4, 8.6, 8.8, 9, 9.2, 9.4, 9.6, 9.8, 10, 10.2, 10.4, 10.6])
waveDirections = np.array([np.pi])

# mesh resolution
panelsPerMeter = 65

# number of modal components considered
numberModes = 6



# CALCULATIONS

# old beam definition
hullDraft = hullDisplacement / (waterDensity * hullLength * hullBreadth)
zNeutralAxis = -hullDraft + zCentroidOverBottom
zTwistCenter = -hullDraft + zShearCenterOverBottom
centerOfMass = (hullLength/2, 0, zNeutralAxis)

uniformlyDistributedMass = hullDisplacement - np.sum(pointMasses)
linearDensitiesBeam = np.ones([beamSegments]) * uniformlyDistributedMass / hullLength

beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(0, hullLength, beamSegments + 1)
beamDefinition['crossSectionAreas'] = sectionalAreas
beamDefinition['verticalAreaMoments'] = verticalAreaMoments
beamDefinition['horizontalAreaMoments'] = horizontalAreaMoments
beamDefinition['verticalTimoshenkoCoefs'] = verticalShearAreaFractions
beamDefinition['horizontalTimoshenkoCoefs'] = horizontalShearAreaFractions
beamDefinition['torsionConstants'] = torsionConstants
beamDefinition['warpingConstants'] = warpingConstants
beamDefinition['youngsModulus'] = youngsModulus
beamDefinition['shearModulus'] = shearModulus
beamDefinition['zNeutralAxis'] = zNeutralAxis
beamDefinition['zTwistCenter'] = zTwistCenter
beamDefinition['linearDensities'] = linearDensitiesBeam
beamDefinition['zCentersOfMass'] = np.ones([beamSegments]) * zNeutralAxis
beamDefinition['rollInertias'] = linearDensitiesBeam * (hullBreadth*0.35)**2

beam = spr.Beam(beamDefinition)

if not beamSegments % 40 == 0:
    sys.exit('Beam must be discretized in a number of segments that is a multiple of 40.')
segmentsPerHalfStation = beamSegments / 40
for i in range(pointMasses.size):
    vertex = int(pointMassStations[i] * 2 * segmentsPerHalfStation)

    beam.massMatrix[7 * vertex    , 7 * vertex    ] += pointMasses[i]
    beam.massMatrix[7 * vertex + 1, 7 * vertex + 1] += pointMasses[i]
    beam.massMatrix[7 * vertex + 2, 7 * vertex + 2] += pointMasses[i]

    beam.massMatrix[7 * vertex + 3, 7 * vertex + 3] += pointMasses[i] * (zNeutralAxis - zTwistCenter)**2

    beam.massMatrix[7 * vertex + 1, 7 * vertex + 3] -= pointMasses[i] * (zNeutralAxis - zTwistCenter)
    beam.massMatrix[7 * vertex + 3, 7 * vertex + 1] -= pointMasses[i] * (zNeutralAxis - zTwistCenter)

# new beam definition
pointMassXPositions = pointMassStations * hullLength/20
pointMassPositions = np.zeros([pointMasses.size, 3])
pointMassPositions[:, 0] = pointMassXPositions
pointMassPositions[:, 2] = zNeutralAxis

beamDefinition['pointMasses'] = pointMasses
beamDefinition['pointMassPositions'] = pointMassPositions

newBeam = spr.Beam(beamDefinition)



# OUTPUT

print('Max error:')
print(np.max(beam.massMatrix - newBeam.massMatrix))