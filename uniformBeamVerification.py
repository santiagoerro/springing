import numpy as np
import numpy.linalg as la
import springing as spr
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp
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

# torsion-bending coupling
axesOffset = 1

# number of modes
numberModes = 10



# CALCULATIONS
# beam definition without bending-torsion coupling
shearModulus = youngsModulus / (2 * (1 + poissonRatio))

verticalTimoshenkoCoef = 3 * youngsModulus * verticalAreaMoment / (vertical3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)
horizontalTimoshenkoCoef = 3 * youngsModulus * horizontalAreaMoment / (horizontal3EIOverKappaLSquaredAG * sectionArea * shearModulus * beamLength**2)

warpingConstant = shearModulus * torsionConstant / (youngsModulus * warpingWavenumberBeamLength**2) * beamLength**2

meshType = 'uniform'
if meshType == 'uniform':
    nodeXPositions = np.linspace(0, beamLength, beamSegments + 1)
else:
    nodeXPositions = np.linspace(0, beamLength, beamSegments)
    nodeXPositions = np.sort(np.append(nodeXPositions, beamLength/(beamSegments - 1) / 2))

sectionalAreas = np.ones([beamSegments]) * sectionArea
verticalAreaMoments = np.ones([beamSegments]) * verticalAreaMoment
horizontalAreaMoments = np.ones([beamSegments]) * horizontalAreaMoment
verticalTimoshenkoCoefs = np.ones([beamSegments]) * verticalTimoshenkoCoef
horizontalTimoshenkoCoefs = np.ones([beamSegments]) * horizontalTimoshenkoCoef
torsionConstants = np.ones([beamSegments]) * torsionConstant
warpingConstants = np.ones([beamSegments]) * warpingConstant
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
beamDefinition['zNeutralAxis'] = 0
beamDefinition['zTwistCenter'] = 0
beamDefinition['linearDensities'] = linearDensities
beamDefinition['zCentersOfMass'] = zCentersOfMass
beamDefinition['rollInertias'] = rollInertias

beam = spr.Beam(beamDefinition)


# numerical natural frequencies
dryNaturalFrequenciesSquared, dryVibrationModesNormalized = beam.CalculateModes(7 * (beamSegments + 1), rigidBodyModesFrequencySquaredTolerance = 1e-2)
dryNaturalFrequencies = np.sqrt(dryNaturalFrequenciesSquared)

# mode classification
axialDryNaturalFrequencies = np.zeros([numberModes])
verticalBendingDryNaturalFrequencies = np.zeros([numberModes])
horizontalBendingDryNaturalFrequencies = np.zeros([numberModes])
torsionDryNaturalFrequencies = np.zeros([numberModes])

axialCounter = 0
verticalBendingCounter = 0
horizontalBendingCounter = 0
torsionCounter = 0

for i in range(6, 7 * (beamSegments + 1)):
    if np.abs(dryVibrationModesNormalized[0, i]) > 0.1 and axialCounter < 10:
        axialDryNaturalFrequencies[axialCounter] = dryNaturalFrequencies[i]
        axialCounter = axialCounter + 1
    elif np.abs(dryVibrationModesNormalized[2, i]) > 0.1 and verticalBendingCounter < 10:
        verticalBendingDryNaturalFrequencies[verticalBendingCounter] = dryNaturalFrequencies[i]
        verticalBendingCounter = verticalBendingCounter + 1
    elif np.abs(dryVibrationModesNormalized[1, i]) > 0.1 and horizontalBendingCounter < 10:
        horizontalBendingDryNaturalFrequencies[horizontalBendingCounter] = dryNaturalFrequencies[i]
        horizontalBendingCounter = horizontalBendingCounter + 1
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
horizontalBendingDryNaturalFrequenciesBernoulliAnalytic = betaSolutions**2 / beamLength**2 * np.sqrt(youngsModulus * horizontalAreaMoments[0] / linearDensities[0])

torsionalDryNaturalFrequenciesNoWarpingAnalytic = np.arange(1, numberModes + 1) * np.pi / beamLength * np.sqrt(shearModulus * torsionConstants[0] / rollInertias[0])


# internal force distributions and displacement functions
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
x = np.linspace(0, beamLength, 5000)
verticalBendingMoment = beam.InternalForce(x, nodalDisplacementsBending, 'mv')
verticalShearForce = beam.InternalForce(x, nodalDisplacementsBending, 'sv')
torsionMoment = beam.InternalForce(x, nodalDisplacementsTorsion, 't')
torsionMomentFree = beam.InternalForce(x, nodalDisplacementsTorsion, 'tf')
torsionMomentWarping = beam.InternalForce(x, nodalDisplacementsTorsion, 'tw')
# displacement functions
verticalDeflection = beam.DisplacementFunction(x, nodalDisplacementsBending, 'v')
twist = beam.DisplacementFunction(x, nodalDisplacementsTorsion, 't')


# beam definition with torsion-bending coupling
beamDefinitionCoupled = beamDefinition
beamDefinitionCoupled['zTwistCenter'] = -axesOffset

beamCoupled = spr.Beam(beamDefinitionCoupled)

# FEM natural frequencies
dryNaturalFrequenciesCoupledSquared, dryVibrationModesCoupledNormalized = beamCoupled.CalculateModes(7 * (beamSegments + 1))
dryNaturalFrequenciesCoupled = np.sqrt(dryNaturalFrequenciesCoupledSquared)

# mode classification
torsionBendingDryNaturalFrequencies = np.zeros([numberModes])
torsionBendingDryVibrationModesNormalized = np.zeros([7 * (beamSegments + 1), numberModes])

torsionBendingCounter = 0

for i in range(6, 7 * (beamSegments + 1)):
    if np.abs(dryVibrationModesCoupledNormalized[3, i]) > 0.1 and torsionBendingCounter < 10:
        torsionBendingDryNaturalFrequencies[torsionBendingCounter] = dryNaturalFrequenciesCoupled[i]
        torsionBendingDryVibrationModesNormalized[:, torsionBendingCounter] = dryVibrationModesCoupledNormalized[:, i]
        torsionBendingCounter = torsionBendingCounter + 1

# boundary value problem solution of continuous coupled Euler-Bernoulli bending and warping-free torsion
def BendingTorsionODEFunction(x: np.ndarray, y: np.ndarray, p: np.ndarray):
    h = y[0, :]
    hPrime = y[1, :]
    hDoublePrime = y[2, :]
    hTriplePrime = y[3, :]
    alpha = y[4, :]
    alphaPrime = y[5, :]

    omegaSquared = p[0]

    yPrime = np.zeros(y.shape)
    yPrime[0, :] = hPrime
    yPrime[1, :] = hDoublePrime
    yPrime[2, :] = hTriplePrime
    yPrime[3, :] = omegaSquared * linearDensity / (youngsModulus * horizontalAreaMoment) * (h - axesOffset * alpha)
    yPrime[4, :] = alphaPrime
    yPrime[5, :] = omegaSquared / (shearModulus * torsionConstant) * (axesOffset * linearDensity * h - (axesOffset**2 * linearDensity + rollInertia) * alpha)

    return yPrime


def BendingTorsionBC(yInitial: np.ndarray, yFinal: np.ndarray, p: np.ndarray):
    boundaryConditions = np.zeros([7])

    boundaryConditions[0] = yInitial[2]
    boundaryConditions[1] = yInitial[3]
    boundaryConditions[2] = yInitial[5]
    boundaryConditions[3] = yFinal[2]
    boundaryConditions[4] = yFinal[3]
    boundaryConditions[5] = yFinal[5]
    boundaryConditions[6] = yInitial[4] + 1

    return boundaryConditions


torsionBendingDryNaturalFrequenciesBVP = np.zeros([numberModes])

for i in range(numberModes):
    yGuess = np.zeros([6, nodeXPositions.size])

    yGuess[0, :] = torsionBendingDryVibrationModesNormalized[1::7, i]
    yGuess[1, :] = np.gradient(yGuess[0, :], nodeXPositions)
    yGuess[2, :] = np.gradient(yGuess[1, :], nodeXPositions)
    yGuess[3, :] = np.gradient(yGuess[2, :], nodeXPositions)
    yGuess[4, :] = torsionBendingDryVibrationModesNormalized[3::7, i]
    yGuess[5, :] = np.gradient(yGuess[4, :], nodeXPositions)

    pGuess = np.array([torsionBendingDryNaturalFrequencies[i]**2])

    solution = solve_bvp(BendingTorsionODEFunction, BendingTorsionBC, nodeXPositions, yGuess, pGuess)

    torsionBendingDryNaturalFrequenciesBVP[i] = np.sqrt(solution.p[0])



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
print('Dry horizontal bending natural frequencies (rad/s)')
print('Nodes     Numerical     Analytic Bernoulli beam')
for i in range(numberModes):
    print('%2d        %6.2f        %6.2f'%(i + 2, horizontalBendingDryNaturalFrequencies[i], horizontalBendingDryNaturalFrequenciesBernoulliAnalytic[i]))
print()
print('Dry decoupled torsion natural frequencies (rad/s)')
print('Nodes     Numerical     Analytic no warping')
for i in range(numberModes):
    print('%2d        %5.2f         %5.2f'%(i + 2, torsionDryNaturalFrequencies[i], torsionalDryNaturalFrequenciesNoWarpingAnalytic[i]))
print()
print('Dry coupled horizontal bending and torsion natural frequencies (rad/s)')
print('Number    FEM           BVP Bernoulli no warping')
for i in range(numberModes):
    print('%2d        %5.2f         %5.2f'%(i + 1, torsionBendingDryNaturalFrequencies[i], torsionBendingDryNaturalFrequenciesBVP[i]))
print()


plt.figure()
plt.title('Vertical deflection distribution clamped start, 1N point force end.')
plt.axhline(color = 'k', linewidth = 1)
plt.plot(x, verticalDeflection, 'b')
plt.xlabel('x [m]')
plt.ylabel('Vertical deflection [m]')

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
plt.title('Twist angle distribution clamped start, 1Nm point moment end.')
plt.axhline(color = 'k', linewidth = 1)
plt.plot(x, twist, 'b')
plt.xlabel('x [m]')
plt.ylabel('Twist angle [rad]')

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