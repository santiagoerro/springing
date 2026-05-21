import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import springing as spr
import capytaine as cpt



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
vertical3EIOverKappaLSquaredAG = 0.002
horizontal3EIOverKappaLSquaredAG = 0.002

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

meshSize = (beamLength, beamLength * 0.1, beamLength * 0.1)
meshCenter = (beamLength/2, 0, 0)
meshResolution = (100, 10, 10)

mesh = cpt.mesh_parallelepiped(size = meshSize, center = meshCenter, name = 'hull', resolution = meshResolution).immersed_part()

beam = spr.Beam(beamDefinition)

dryNaturalFrequenciesSquared, dryVibrationModesNormalized, modalDofs = beam.CalculateModalDOFs(mesh, 100)


def BendingTorsionODEFunction(x: np.ndarray, y: np.ndarray, p: np.ndarray):
    h = y[0, :]
    hPrime = y[1, :]
    hDoublePrime = y[2, :]
    hTriplePrime = y[3, :]
    a = y[4, :]
    aPrime = y[5, :]

    omegaSquared = p[0]

    yPrime = np.zeros(y.shape)
    yPrime[0, :] = hPrime
    yPrime[1, :] = hDoublePrime
    yPrime[2, :] = hTriplePrime
    yPrime[3, :] = omegaSquared * linearDensity / (youngsModulus * horizontalAreaMoment) * (h - axesOffset * a)
    yPrime[4, :] = aPrime
    yPrime[5, :] = omegaSquared / (shearModulus * torsionConstant) * (axesOffset * linearDensity * h - (axesOffset**2 * linearDensity + rollInertiaDensity) * a)

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


xGuess = np.linspace(0, beamLength, initialNodes)
yGuess = np.zeros([6, initialNodes])

bendingCoef = bendingNodes * np.pi / beamLength
torsionCoef = torsionNodes * np.pi / beamLength

yGuess[0, :] = -axesOffset * np.cos(bendingCoef * xGuess)
yGuess[1, :] = axesOffset * bendingCoef * np.sin(bendingCoef * xGuess)
yGuess[2, :] = axesOffset * bendingCoef**2 * np.cos(bendingCoef * xGuess)
yGuess[3, :] = -axesOffset * bendingCoef**3 * np.sin(bendingCoef * xGuess)
yGuess[4, :] = -np.cos(torsionCoef * xGuess)
yGuess[5, :] = torsionCoef * np.sin(torsionCoef * xGuess)

hQuadruplePrimeGuess = -axesOffset * bendingCoef**4 * np.cos(bendingCoef * xGuess)
hQuadruplePrimeODEFromGuessOverOmegaSquared = linearDensity / (youngsModulus * horizontalAreaMoment) * (yGuess[0, :] - axesOffset * yGuess[4, :])

aDoublePrimeGuess = torsionCoef**2 * np.cos(torsionCoef * xGuess)
aDoublePrimeODEFromGuessOverOmegaSquared = 1 / (shearModulus * torsionConstant) * (axesOffset * linearDensity * yGuess[0, :] - (axesOffset**2 * linearDensity + rollInertiaDensity) * yGuess[4, :])

omegaSquaredGuess = np.sum(aDoublePrimeGuess * aDoublePrimeODEFromGuessOverOmegaSquared) / np.sum(aDoublePrimeODEFromGuessOverOmegaSquared**2)

solution = solve_bvp(BendingTorsionODEFunction, BendingTorsionBC, xGuess, yGuess, np.array([omegaSquaredGuess]))

ySolution = solution.sol(xGuess)



print('Omega squared guess:')
print(omegaSquaredGuess)
print('Omega squared:')
print(solution.p[0])

plt.figure()
plt.plot(xGuess, hQuadruplePrimeGuess)
plt.plot(xGuess, hQuadruplePrimeODEFromGuessOverOmegaSquared, '--')

plt.figure()
plt.plot(xGuess, aDoublePrimeGuess)
plt.plot(xGuess, aDoublePrimeODEFromGuessOverOmegaSquared, '--')

plt.figure()
plt.plot(xGuess, yGuess[0, :], 'b--')
plt.plot(xGuess, yGuess[4, :], 'g--')
plt.plot(xGuess, ySolution[0, :])
plt.plot(xGuess, ySolution[4, :], '--')

plt.show()