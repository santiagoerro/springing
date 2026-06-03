import numpy as np
import springing as spr
import capytaine as cpt
import time as tm



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

tic = tm.time()
nodalDofsOld = beam.CalculateNodalDOFs(mesh)
tac = tm.time()
elapsedTimeOldFunction = tac - tic

tic = tm.time()
nodalDofsNew = beam.CalculateNodalDOFsNew(mesh)
tac = tm.time()
elapsedTimeNewFunction = tac - tic

maxError = 0

for dofName in nodalDofsOld.keys():
    errors = np.abs(nodalDofsNew[dofName] - nodalDofsOld[dofName])
    if np.max(errors) > maxError:
        maxError = np.max(errors)

print('Old function: %f s'%elapsedTimeOldFunction)
print('New function: %f s'%elapsedTimeNewFunction)
print('Max error: %e'%maxError)