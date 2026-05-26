import numpy as np
import matplotlib.pyplot as plt
import springing as spr



linearDensity = 1
youngsModulus = 1
horizontalAreaMoment = 1
axesOffset = 1
shearModulus = 1
torsionConstant = 1
rollInertiaDensity = 1

beamLength = 1

bendingNodes = 2
torsionNodes = 6
initialNodes = 100


verticalAreaMoment = 1
sectionArea = 1
vertical3EIOverKappaLSquaredAG = 0.00002
horizontal3EIOverKappaLSquaredAG = 0.00002

warpingWavenumberBeamLength = 80

numberSegments = 40

verticalTimoshenkoCoef = 3 * youngsModulus * verticalAreaMoment / (vertical3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)
horizontalTimoshenkoCoef = 3 * youngsModulus * horizontalAreaMoment / (horizontal3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)

warpingConstant = shearModulus * torsionConstant / (youngsModulus * warpingWavenumberBeamLength**2) * beamLength**2


beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(0, beamLength, numberSegments + 1)
beamDefinition['crossSectionAreas'] = np.ones([numberSegments]) * sectionArea
beamDefinition['verticalAreaMoments'] = np.ones([numberSegments]) * verticalAreaMoment
beamDefinition['horizontalAreaMoments'] = np.ones([numberSegments]) * horizontalAreaMoment
beamDefinition['verticalTimoshenkoCoefs'] = np.ones([numberSegments]) * verticalTimoshenkoCoef
beamDefinition['horizontalTimoshenkoCoefs'] = np.ones([numberSegments]) * horizontalTimoshenkoCoef
beamDefinition['torsionConstants'] = np.ones([numberSegments]) * torsionConstant
beamDefinition['warpingConstants'] = np.ones([numberSegments]) * warpingConstant
beamDefinition['youngsModulus'] = youngsModulus
beamDefinition['shearModulus'] = shearModulus
beamDefinition['zNeutralAxis'] = 0
beamDefinition['zTwistCenter'] = -axesOffset
beamDefinition['linearDensities'] = np.ones([numberSegments]) * linearDensity
beamDefinition['zCentersOfMass'] = np.zeros([numberSegments])
beamDefinition['rollInertias'] = np.ones([numberSegments]) * rollInertiaDensity


beam = spr.Beam(beamDefinition)


x = np.linspace(-beamLength * 0.1, beamLength * 1.1, 5000)

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 0] = 1
axialDisplacementFunction = beam.DisplacementFunction(x, displacements, 'a')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 1] = 1
horizontalDeflectionDisplacementFunctionSwayMotion = beam.DisplacementFunction(x, displacements, 'h')
yawDisplacementFunctionSwayMotion = beam.DisplacementFunction(x, displacements, 'q')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 6] = 1
horizontalDeflectionDisplacementFunctionYawMotion = beam.DisplacementFunction(x, displacements, 'h')
yawDisplacementFunctionYawMotion = beam.DisplacementFunction(x, displacements, 'q')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 2] = 1
verticalDeflectionDisplacementFunctionHeaveMotion = beam.DisplacementFunction(x, displacements, 'v')
pitchDisplacementFunctionHeaveMotion = beam.DisplacementFunction(x, displacements, 'p')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 5] = 1
verticalDeflectionDisplacementFunctionPitchMotion = beam.DisplacementFunction(x, displacements, 'v')
pitchDisplacementFunctionPitchMotion = beam.DisplacementFunction(x, displacements, 'p')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 3] = 1
twistDisplacementFunctionPhiMotion = beam.DisplacementFunction(x, displacements, 't')

displacements = np.zeros([7 * (numberSegments + 1)])
displacements[7 * 3 + 4] = 1
twistDisplacementFunctionWarpingMotion = beam.DisplacementFunction(x, displacements, 't')



plt.figure()
plt.title('Axial motion')
plt.plot(x, axialDisplacementFunction)

plt.figure()
plt.title('Sway motion deflection')
plt.plot(x, horizontalDeflectionDisplacementFunctionSwayMotion)

plt.figure()
plt.title('Sway motion rotation')
plt.plot(x, yawDisplacementFunctionSwayMotion)

plt.figure()
plt.title('Yaw motion deflection')
plt.plot(x, horizontalDeflectionDisplacementFunctionYawMotion)

plt.figure()
plt.title('Yaw motion rotation')
plt.plot(x, yawDisplacementFunctionYawMotion)

plt.figure()
plt.title('Heave motion deflection')
plt.plot(x, verticalDeflectionDisplacementFunctionHeaveMotion)

plt.figure()
plt.title('Heave motion rotation')
plt.plot(x, pitchDisplacementFunctionHeaveMotion)

plt.figure()
plt.title('Pitch motion deflection')
plt.plot(x, verticalDeflectionDisplacementFunctionPitchMotion)

plt.figure()
plt.title('Pitch motion rotation')
plt.plot(x, pitchDisplacementFunctionPitchMotion)

plt.figure()
plt.title('Phi motion')
plt.plot(x, twistDisplacementFunctionPhiMotion)

plt.figure()
plt.title('Warping motion')
plt.plot(x, twistDisplacementFunctionWarpingMotion)

plt.show()