import numpy as np
import numpy.linalg as la
import capytaine as cpt
import springing as spr
import xarray as xr
from matplotlib import pyplot as plt

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
omegas = np.array([6.07])
waveDirections = np.array([np.pi])

# mesh resolution
panelsPerMeter = 65

# detailed analysis frequency
omegaDetail = 6.07



# CALCULATIONS

hullDraft = hullDisplacement / (waterDensity * hullLength * hullBreadth)
zNeutralAxis = -hullDraft + zCentroidOverBottom
uniformlyDistributedMass = hullDisplacement - np.sum(pointMasses)
pointMassXPositions = pointMassStations * hullLength/20 - hullLength/2

# mesh generation
panelsLength = int(round(panelsPerMeter * hullLength))
panelsBreadth = int(round(panelsPerMeter * hullBreadth))
panelsDepth = int(round(panelsPerMeter * hullDepth))

hullSize = (hullLength, hullBreadth, hullDepth)
hullCenter = (0, 0, -hullDraft + hullDepth/2)
meshResolution = (panelsLength, panelsBreadth, panelsDepth)

hullMesh: cpt.Mesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()


# definition of the body with only rigid body dofs
xCenterOfMass = np.sum(pointMassXPositions * pointMasses) / hullDisplacement
centerOfMass = (xCenterOfMass, 0, zNeutralAxis)

hullRigidBody = cpt.FloatingBody(mesh = hullMesh, dofs = cpt.rigid_body_dofs(rotation_center = centerOfMass), center_of_mass = centerOfMass)

# hydrostatic stiffness calculation for rigid body
hydrostaticStiffness = hullRigidBody.compute_hydrostatic_stiffness(rho = waterDensity, g = gravity)

# rigid body mass matrix
xInertiaContibution = 1/12 * uniformlyDistributedMass * hullLength**2 + np.sum((pointMassXPositions - xCenterOfMass)**2 * pointMasses)
rollInertia = uniformlyDistributedMass * (hullBreadth*0.35)**2

massMatrix = np.zeros([6,6])
massMatrix[0, 0] = hullDisplacement
massMatrix[1, 1] = hullDisplacement
massMatrix[2, 2] = hullDisplacement
massMatrix[3, 3] = rollInertia
massMatrix[4, 4] = xInertiaContibution
massMatrix[5, 5] = xInertiaContibution
massMatrix = xr.DataArray(massMatrix, dims = ['influenced_dof', 'radiating_dof'])

# rigid body hydrodynamic calculation: added mass, radiation, forcing
testMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(hullRigidBody.dofs),
    'water_depth': waterDepths,
    'rho': waterDensity,
    'g': gravity
})

hydrodynamicSeakeepingResults = cpt.BEMSolver().fill_dataset(testMatrix, hullRigidBody)

omegaIndex = np.argmin(np.abs(omegas - omegaDetail))
diffractionProblem = cpt.DiffractionProblem(body = hullRigidBody, omega = omegas[omegaIndex], wave_direction = waveDirections[0], water_depth = waterDepths[0], rho = waterDensity, g = gravity)
diffractionResults = cpt.BEMSolver().solve(diffractionProblem, keep_details = True)
froudeKrylovPressures = cpt.bem.airy_waves.airy_waves_pressure(hullMesh.faces_centers, diffractionProblem)
radiationResults = np.zeros([6], dtype = object)
for i in range(6):
    dofName = list(hullRigidBody.dofs)[i]
    radiationProblem = cpt.RadiationProblem(body = hullRigidBody, omega = omegas[omegaIndex], water_depth = waterDepths[0], rho = waterDensity, g = gravity, radiating_dof = dofName)
    radiationResults[i] = cpt.BEMSolver().solve(radiationProblem, keep_details = True)

# coupling of hydrodynamics and rigid body, seakeeping results
forcesFromAmplitudesMatrices: xr.DataArray = - (hydrodynamicSeakeepingResults.added_mass + massMatrix) * hydrodynamicSeakeepingResults.omega**2 + complex(0,1) * hydrodynamicSeakeepingResults.omega * hydrodynamicSeakeepingResults.radiation_damping + hydrostaticStiffness

amplitudesFromForcesMatrices = xr.DataArray(la.inv(forcesFromAmplitudesMatrices), dims = ['omega', 'influenced_dof', 'radiating_dof'])

seakeepingDisplacements: xr.DataArray = xr.dot(hydrodynamicSeakeepingResults.excitation_force, amplitudesFromForcesMatrices, dims = ['influenced_dof'])
seakeepingDisplacements = seakeepingDisplacements.rename({'radiating_dof': 'dof'})

# seakeeping responses
heaveAmplitudesSeakeeping = np.abs(waveHeight / 2 * seakeepingDisplacements[:, 0, 2])
pitchAmplitudesSeakeeping = np.abs(waveHeight / 2 * seakeepingDisplacements[:, 0, 4])

# wavelengths
wavelengths = hydrodynamicSeakeepingResults.wavelength.values

# integration of seakeeping pressures to compute shear forces and bending moments
# Froude-Krylov, diffraction and radiation pressures are integrated. Hydrostatics can be ignored ONLY for a rectangular waterplane surface.
xVerticalLoads = np.linspace(-hullLength/2 + hullLength/panelsLength/2, hullLength/2 - hullLength/panelsLength/2, panelsLength)
verticalExcitationLoads = np.zeros_like(xVerticalLoads, dtype = froudeKrylovPressures.dtype)
verticalRadiationLoads = np.zeros_like(xVerticalLoads, dtype = froudeKrylovPressures.dtype)

for i in range(hullMesh.nb_faces):
    xPositionIndex = np.argmin(np.abs(xVerticalLoads - hullMesh.faces_centers[i, 0]))

    verticalExcitationLoads[xPositionIndex] -= waveHeight/2 * froudeKrylovPressures[i] * hullMesh.faces_areas[i] * hullMesh.faces_normals[i, 2] / (hullLength/panelsLength)
    verticalExcitationLoads[xPositionIndex] -= waveHeight/2 * diffractionResults.pressure[i] * hullMesh.faces_areas[i] * hullMesh.faces_normals[i, 2] / (hullLength/panelsLength)
    for j in range(6):
        verticalRadiationLoads[xPositionIndex] -= waveHeight / 2 * seakeepingDisplacements.values[omegaIndex, 0, j] * radiationResults[j].pressure[i] * hullMesh.faces_areas[i] * hullMesh.faces_normals[i, 2] / (hullLength/panelsLength)

def ShearForceAndBendingMomentFromVerticalLoads(verticalLoads: np.ndarray):
    verticalLoads -= np.sum(verticalLoads * xVerticalLoads) / np.sum(xVerticalLoads**2) * xVerticalLoads
    verticalLoads -= np.mean(verticalLoads)

    xForceDistributions = np.linspace(-hullLength/2, hullLength/2, panelsLength + 1)
    shearForces = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)

    for i in range(1, xForceDistributions.size):
        shearForces[i] = shearForces[i - 1] - verticalLoads[i - 1] * hullLength/panelsLength

    bendingMoments = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)

    for i in range(1, xForceDistributions.size):
        bendingMoments[i] = bendingMoments[i - 1] + 0.5 * (shearForces[i - 1] + shearForces[i]) * hullLength/panelsLength
    
    return xForceDistributions, shearForces, bendingMoments

xForceDistributions, excitationShearForces, excitationBendingMoments = ShearForceAndBendingMomentFromVerticalLoads(verticalExcitationLoads)
_, radiationShearForces, radiationBendingMoments = ShearForceAndBendingMomentFromVerticalLoads(verticalRadiationLoads)

shearForces = excitationShearForces + radiationShearForces
bendingMoments = excitationBendingMoments + radiationBendingMoments

midshipsBendingMoment = np.abs(bendingMoments[int(round(bendingMoments.size/2))])
midshipsBendingMomentCoef = midshipsBendingMoment / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)


# springing beam definition
zTwistCenter = -hullDraft + zShearCenterOverBottom
linearDensitiesBeam = np.ones([beamSegments]) * uniformlyDistributedMass / hullLength
pointMassPositions = np.zeros([pointMasses.size, 3])
pointMassPositions[:, 0] = pointMassXPositions
pointMassPositions[:, 2] = zNeutralAxis

beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(-hullLength/2, hullLength/2, beamSegments + 1)
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
beamDefinition['pointMasses'] = pointMasses
beamDefinition['pointMassPositions'] = pointMassPositions

beam = spr.Beam(beamDefinition)

# springing modal dofs
dryNaturalFrequenciesSquared, dryVibrationModesNormalized, modalDofs = beam.CalculateModalDOFs(hullMesh, (beamSegments + 1) * 7)

# definition of the springing body
hullBodyModalSpringing = cpt.FloatingBody(mesh = hullMesh, dofs = modalDofs, center_of_mass = centerOfMass)

# springing hydrodynamic calculation
testMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(hullBodyModalSpringing.dofs)[:6],
    'water_depth': waterDepths,
    'rho': waterDensity,
    'g': gravity
})

hydrodynamicModalSpringingResults = cpt.BEMSolver().fill_dataset(testMatrix, hullBodyModalSpringing)

# rigid body hydrostatics into modal springing
modalHydrostaticStiffness = spr.ComputeHydrostaticStiffness(hullBodyModalSpringing, waterDensity, gravity)

# springing results
modalSpringingResults = spr.ModalProperSpringingResults(dryNaturalFrequenciesSquared, modalHydrostaticStiffness, hydrodynamicModalSpringingResults)

# springing heave and pitch responses
heaveAmplitudesSpringing = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes[:, 0, 2] * dryVibrationModesNormalized[2, 2])
pitchAmplitudesSpringing = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes[:, 0, 4] * dryVibrationModesNormalized[5, 4])

# midships bending moments
midshipsBendingMomentsSpringing = np.zeros([omegas.size])

for i in range(omegas.size):
    displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[i, 0, :]
    midshipsBendingMomentsSpringing[i] = np.abs(beam.InternalForce(hullLength/2, displacements, 'mv'))

bendingMomentCoefsSpringing = midshipsBendingMomentsSpringing / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)

# bending moment and shear force distributions
x = np.linspace(0, hullLength, 500)
displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[omegaIndex, 0, :]
springingBendingMoments = beam.InternalForce(x, displacements, 'mv')
springingShearForces = beam.InternalForce(x, displacements, 'sv')



# OUTPUT

print('Midships bending moment, seakeeping: %.2f Nm'%midshipsBendingMoment)
print('Midships bending moment, springing: %.2f Nm'%midshipsBendingMomentsSpringing[omegaIndex])
print('Midships bending moment coefficient, seakeeping: %.4f '%midshipsBendingMomentCoef)
print('Midships bending moment coefficient, springing: %.4f '%bendingMomentCoefsSpringing[omegaIndex])
print()
print('Heave RAO')
print('Ship length / wavelength: %.2f'%(hullLength / wavelengths[omegaIndex]))
print('Heave RAO seakeeping: %.4f m/m'%(heaveAmplitudesSeakeeping[omegaIndex] / (waveHeight / 2)))
print('Heave RAO springing: %.4f m/m'%(heaveAmplitudesSpringing[omegaIndex] / (waveHeight / 2)))
print()
print('Pitch RAO')
print('Ship length / wavelength: %.2f'%(hullLength / wavelengths[omegaIndex]))
print('Pitch RAO seakeeping: %.4f rad/rad'%(pitchAmplitudesSeakeeping[omegaIndex] / (waveHeight / wavelengths[omegaIndex] * np.pi)))
print('Pitch RAO springing: %.4f rad/rad'%(pitchAmplitudesSpringing[omegaIndex] / (waveHeight / wavelengths[omegaIndex] * np.pi)))
print()


plt.figure()
plt.title('Vertical bending moment distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(xForceDistributions + hullLength/2, np.imag(bendingMoments), 'g', label = 'Imaginary part, manual integration of pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(bendingMoments), 'k', label = 'Real part, manual integration of pressures')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationBendingMoments), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationBendingMoments), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.plot(x, np.imag(springingBendingMoments), 'g:', label = 'Imaginary part, springing')
plt.plot(x, np.real(springingBendingMoments), 'k:', label = 'Real part, springing')
plt.xlabel('x [m]')
plt.ylabel('Vertical bending moment [Nm]')
plt.legend()

plt.figure()
plt.title('Vertical shear force distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(xForceDistributions + hullLength/2, np.imag(shearForces), 'g', label = 'Imaginary part, manual integration of pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(shearForces), 'k', label = 'Real part, manual integration of pressures')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationShearForces), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationShearForces), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.plot(x, np.imag(springingShearForces), 'g:', label = 'Imaginary part, springing')
plt.plot(x, np.real(springingShearForces), 'k:', label = 'Real part, springing')
plt.xlabel('x [m]')
plt.ylabel('Vertical shear force [N]')
plt.legend()

plt.show()