import numpy as np
import numpy.linalg as la
import springing as spr
import capytaine as cpt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt



# INPUT
# beam properties
beamLength = 1
beamSegments = 40

# material properties
youngsModulus = 1
poissonRatio = 0.26

# cross sectional properties
sectionArea = 1
verticalAreaMoment = 1
horizontalAreaMoment = 2
torsionConstant = 1

vertical3EIOverKappaLSquaredAG = 0.002
horizontal3EIOverKappaLSquaredAG = 0.002

warpingWavenumberBeamLength = 80

# mass properties
linearDensity = 1
rollInertia = 1

# number of modes
numberModes = 10



# CALCULATIONS
# beam definition
shearModulus = youngsModulus / (2 * (1 + 0.26))

verticalTimoshenkoCoef = 3 * youngsModulus * verticalAreaMoment / (vertical3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)
horizontalTimoshenkoCoef = 3 * youngsModulus * horizontalAreaMoment / (horizontal3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)

warpingConstant = shearModulus * torsionConstant / (youngsModulus * warpingWavenumberBeamLength**2) * beamLength**2

nodeXPositions = np.linspace(0, beamLength, beamSegments + 1)
sectionalAreas = np.ones([beamSegments]) * sectionArea
verticalAreaMoments = np.ones([beamSegments]) * verticalAreaMoment
horizontalAreaMoments = np.ones([beamSegments]) * horizontalAreaMoment
verticalTimoshenkoCoefs = np.ones([beamSegments]) * verticalTimoshenkoCoef
horizontalTimoshenkoCoefs = np.ones([beamSegments]) * horizontalTimoshenkoCoef
torsionConstants = np.ones([beamSegments]) * torsionConstant
warpingConstants = np.ones([beamSegments]) * warpingConstant
zNeutralAxis = 0
zTwistCenter = 0
linearDensities = np.ones([beamSegments]) * linearDensity
zCentersOfMass = np.zeros([beamSegments])
rollInertias = np.ones([beamSegments]) * rollInertia

beamDefinition = {}
beamDefinition['nodeXPositions'] = nodeXPositions
beamDefinition['crossSectionAreas'] = sectionalAreas
beamDefinition['verticalAreaMoments'] = verticalAreaMoments
beamDefinition['horizontalAreaMoments'] = horizontalAreaMoments
beamDefinition['verticalTimoshenkoCoefs'] = verticalTimoshenkoCoefs
beamDefinition['horizontalTimoshenkoCoefs'] = horizontalTimoshenkoCoefs
beamDefinition['torsionConstants'] = torsionConstants
beamDefinition['warpingConstants'] = warpingConstants
beamDefinition['youngsModulus'] = youngsModulus
beamDefinition['shearModulus'] = shearModulus
beamDefinition['zNeutralAxis'] = zNeutralAxis
beamDefinition['zTwistCenter'] = zTwistCenter
beamDefinition['linearDensities'] = linearDensities
beamDefinition['zCentersOfMass'] = zCentersOfMass
beamDefinition['rollInertias'] = rollInertias

beam = spr.Beam(beamDefinition)

# mesh
meshSize = (beamLength, beamLength * 0.1, beamLength * 0.1)
meshCenter = (beamLength/2, 0, 0)
meshResolution = (100, 10, 10)

mesh = cpt.mesh_parallelepiped(size = meshSize, center = meshCenter, name = 'hull', resolution = meshResolution).immersed_part()


# numerical natural frequencies
dryNaturalFrequenciesSquared, dryVibrationModesNormalized, modalDofs = beam.CalculateModalDOFs(mesh, 7 * (beamSegments + 1), rigidBodyModesFrequencySquaredTolerance = 1e-2)
dryNaturalFrequencies = np.sqrt(dryNaturalFrequenciesSquared)

# mode classification
axialDryNaturalFrequencies = np.zeros([numberModes])
verticalBendingDryNaturalFrequencies = np.zeros([numberModes])
torsionDryNaturalFrequencies = np.zeros([numberModes])

axialCounter = 0
verticalBendingCounter = 0
torsionCounter = 0

for i in range(6, 7 * (beamSegments + 1) - 6):
    if np.abs(dryVibrationModesNormalized[0, i]) > 0.1 and axialCounter < 10:
        axialDryNaturalFrequencies[axialCounter] = dryNaturalFrequencies[i]
        axialCounter = axialCounter + 1
    elif np.abs(dryVibrationModesNormalized[2, i]) > 0.1 and verticalBendingCounter < 10:
        verticalBendingDryNaturalFrequencies[verticalBendingCounter] = dryNaturalFrequencies[i]
        verticalBendingCounter = verticalBendingCounter + 1
    elif np.abs(dryVibrationModesNormalized[3, i]) > 0.1 and torsionCounter < 10:
        torsionDryNaturalFrequencies[torsionCounter] = dryNaturalFrequencies[i]
        torsionCounter = torsionCounter + 1

# analytic natural frequencies
axialDryNaturalFrequenciesAnalytic = np.arange(1, numberModes + 1) * np.pi / beamLength * np.sqrt(youngsModulus * sectionalAreas[0] / linearDensities[0])

def Equation(beta):
    return np.cosh(beta) * np.cos(beta) - 1

betaSolutions = np.zeros([numberModes])
for i in range(numberModes):
    seed = np.pi * (1.5 + i)
    betaSolutions[i] = fsolve(Equation, seed)[0]

verticalBendingDryNaturalFrequenciesBernoulliAnalytic = betaSolutions**2 / beamLength**2 * np.sqrt(youngsModulus * verticalAreaMoments[0] / linearDensities[0])

torsionalDryNaturalFrequenciesNoWarpingAnalytic = np.arange(1, numberModes + 1) * np.pi / beamLength * np.sqrt(shearModulus * torsionConstants[0] / rollInertias[0])


# internal force distributions
# clamped at initial end
clampedStiffnessMatrix = np.zeros([7 * beamSegments, 7 * beamSegments])
clampedStiffnessMatrix = beam.stiffnessMatrix[7:, 7:]
# 1N point force at final end towards negative z
clampedForcingVectorBending = np.zeros([7 * beamSegments])
clampedForcingVectorBending[7 * (beamSegments - 1) + 2] = -1
clampedNodalDisplacementsBending = la.solve(clampedStiffnessMatrix, clampedForcingVectorBending)
nodalDisplacementsBending = np.zeros([7 * (beamSegments + 1)])
nodalDisplacementsBending[7:] = clampedNodalDisplacementsBending
# 1Nm point torsion moment at final end, right hand rule around x axis
clampedForcingVectorTorsion = np.zeros([7 * beamSegments])
clampedForcingVectorTorsion[7 * (beamSegments - 1) + 3] = 1
clampedNodalDisplacementsTorsion = la.solve(clampedStiffnessMatrix, clampedForcingVectorTorsion)
nodalDisplacementsTorsion = np.zeros([7 * (beamSegments + 1)])
nodalDisplacementsTorsion[7:] = clampedNodalDisplacementsTorsion
# force distributions
x = np.linspace(0, beamLength, 500)
verticalBendingMoment = beam.InternalForce(x, nodalDisplacementsBending, 'mv')
verticalShearForce = beam.InternalForce(x, nodalDisplacementsBending, 'sv')
torsionMoment = beam.InternalForce(x, nodalDisplacementsTorsion, 't')
torsionMomentFree = beam.InternalForce(x, nodalDisplacementsTorsion, 'tf')
torsionMomentWarping = beam.InternalForce(x, nodalDisplacementsTorsion, 'tw')



# OUTPUT
print()
print('Dry axial natural frequencies (rad/s)')
print('Nodes     Numerical     Analytic')
for i in range(numberModes):
    print('%2d        %5.2f         %5.2f'%(i + 1, axialDryNaturalFrequencies[i], axialDryNaturalFrequenciesAnalytic[i]))
print()
print('Dry vertical bending natural frequencies (rad/s)')
print('Nodes     Numerical     Analytic Bernoulli beam')
for i in range(numberModes):
    print('%2d        %6.2f        %6.2f'%(i + 2, verticalBendingDryNaturalFrequencies[i], verticalBendingDryNaturalFrequenciesBernoulliAnalytic[i]))
print()
print('Dry torsion natural frequencies (rad/s)')
print('Nodes     Numerical     Analytic no warping')
for i in range(numberModes):
    print('%2d        %5.2f         %5.2f'%(i + 2, torsionDryNaturalFrequencies[i], torsionalDryNaturalFrequenciesNoWarpingAnalytic[i]))
print()


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

plt.figure()
plt.title('Torsion moment distribution clamped start, 1Nm point moment end.')
plt.axhline(color = 'k', linewidth = 1)
plt.plot(x, torsionMoment, 'b', label = 'Full torsion moment')
plt.plot(x, torsionMomentFree, 'k--', label = 'Free warping component')
plt.plot(x, torsionMomentWarping, 'g--', label = 'Constrained warping component')
plt.xlabel('x [m]')
plt.ylabel('Torsion moment [Nm]')
plt.legend()

plt.show()