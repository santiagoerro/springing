import numpy as np
import capytaine as cpt
import xarray as xr
import os
import springing as spr
from matplotlib import pyplot as plt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.ui.vtk.animation import Animation

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

# detailed analysis frequency
omegaDetail = 6.07



# CALCULATIONS

hullDraft = hullDisplacement / (waterDensity * hullLength * hullBreadth)
zNeutralAxis = -hullDraft + zCentroidOverBottom
zTwistCenter = -hullDraft + zShearCenterOverBottom
centerOfMass = (0, 0, zNeutralAxis)

uniformlyDistributedMass = hullDisplacement - np.sum(pointMasses)
linearDensitiesBeam = np.ones([beamSegments]) * uniformlyDistributedMass / hullLength

pointMassXPositions = pointMassStations * hullLength/20 - hullLength/2
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

# beam is assumed to be parallel to the x axis and oriented towards its positive direction, i.e., the beam normal is [1,0,0]
beam = spr.Beam(beamDefinition)

# mesh generation
panelsLength = int(round(panelsPerMeter * hullLength))
panelsBreadth = int(round(panelsPerMeter * hullBreadth))
panelsDepth = int(round(panelsPerMeter * hullDepth))

hullSize = (hullLength, hullBreadth, hullDepth)
hullCenter = (0, 0, -hullDraft + hullDepth/2)
meshResolution = (panelsLength, panelsBreadth, panelsDepth)

hullMesh: cpt.Mesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()

# creation of dofs from mesh and beam
dryNaturalFrequenciesSquared, dryVibrationModesNormalized, modalDofs = beam.CalculateModalDOFs(hullMesh, (beamSegments + 1) * 7)
dryNaturalFrequenciesHz = np.sqrt(dryNaturalFrequenciesSquared)/(2 * np.pi)

# definition of the body
hullBody = cpt.FloatingBody(mesh = hullMesh, dofs = modalDofs, center_of_mass = centerOfMass)

# solution of the array of problems
testMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(hullBody.dofs)[:numberModes],
    'water_depth': waterDepths,
    'rho': waterDensity,
    'g': gravity
})

# hydrostatic stiffness calculation
hydrostaticStiffness = spr.ComputeHydrostaticStiffness(hullBody, waterDensity, gravity)

# hydrodynamic calculation: added mass, radiation, forcing
hydrodynamicResults = cpt.BEMSolver().fill_dataset(testMatrix, hullBody)
omegaIndex = np.argmin(np.abs(omegas - omegaDetail))
diffractionProblem = cpt.DiffractionProblem(body = hullBody, omega = omegas[omegaIndex], wave_direction = waveDirections[0], water_depth = waterDepths[0], rho = waterDensity, g = gravity)
diffractionResults = cpt.BEMSolver().solve(diffractionProblem, keep_details = True)
froudeKrylovPressures = cpt.bem.airy_waves.airy_waves_pressure(hullMesh.faces_centers, diffractionProblem)

# coupling of hydrodynamic and structural results, springing results
modalSpringingResults = spr.ModalProperSpringingResults(dryNaturalFrequenciesSquared, hydrostaticStiffness, hydrodynamicResults)

# midships bending moments
midshipsBendingMoments = np.zeros([omegas.size], dtype = np.complex128)
midshipsBendingMomentAmplitudes = np.zeros([omegas.size])
heaveAmplitudes = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes.values[:, 0, 2] * dryVibrationModesNormalized[2, 2])
pitchAmplitudes = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes.values[:, 0, 4] * dryVibrationModesNormalized[5, 4])

for i in range(omegas.size):
    displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[i, 0, :]
    midshipsBendingMoments[i] = beam.InternalForce(hullLength/2, displacements, 'mv')
    midshipsBendingMomentAmplitudes[i] = np.abs(midshipsBendingMoments[i])

bendingMomentCoefs = midshipsBendingMomentAmplitudes / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)

# wavelengths
wavelengths = hydrodynamicResults.wavelength.values

# bending moment and shear force distributions
x = np.linspace(0, hullLength, 500)
displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[omegaIndex, 0, :]
bendingMomentDistribution = beam.InternalForce(x, displacements, 'mv')
shearForceDistribution = beam.InternalForce(x, displacements, 'sv')

# force decomposition
froudeKrylovModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
diffractionModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
structuralMassModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
addedMassModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
radiationModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
hydrostaticStiffnessModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
structuralStiffnessModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
matrixTimesAmplitudesModalForces = np.zeros([omegas.size, (beamSegments + 1) * 7], dtype = complex)
modalForcesFromDisplacementsMatrices = np.zeros([omegas.size, (beamSegments + 1) * 7, (beamSegments + 1) * 7], dtype = complex)

for i in range(omegas.size):
    froudeKrylovModalForces[i, :] = hydrodynamicResults.Froude_Krylov_force.values[i, 0, :]
    diffractionModalForces[i, :] = hydrodynamicResults.diffraction_force.values[i, 0, :]

    structuralMassModalForces[i, :] = -omegas[i]**2 * modalSpringingResults.modalAmplitudes.values[i, 0, :]
    addedMassModalForces[i, :] = -omegas[i]**2 * modalSpringingResults.addedMass.values[i, :, :].transpose() @ modalSpringingResults.modalAmplitudes.values[i, 0, :]

    radiationModalForces[i, :] = -complex(0, 1) * omegas[i] * modalSpringingResults.radiationDamping.values[i, :, :].transpose() @ modalSpringingResults.modalAmplitudes.values[i, 0, :]

    hydrostaticStiffnessModalForces[i, :] = hydrostaticStiffness.values @ modalSpringingResults.modalAmplitudes.values[i, 0, :]
    structuralStiffnessModalForces[i, :] = np.diag(dryNaturalFrequenciesSquared) @ modalSpringingResults.modalAmplitudes.values[i, 0, :]

    matrixTimesAmplitudesModalForces[i, :] = modalSpringingResults.modalForcesFromAmplitudesMatrices.values[i, :, :].transpose() @ modalSpringingResults.modalAmplitudes.values[i, 0, :]

    modalForcesFromDisplacementsMatrices[i, :, :] = -omegas[i]**2 * (np.eye((beamSegments + 1) * 7) + modalSpringingResults.addedMass.values[i, :, :].transpose()) - complex(0, 1) * omegas[i] * modalSpringingResults.radiationDamping.values[i, :, :].transpose() + hydrostaticStiffness.values + np.diag(dryNaturalFrequenciesSquared)

def MidshipsBendingMomentFromModalForces(modalForces: np.ndarray):
    elasticModalDisplacements = np.diag(1 / dryNaturalFrequenciesSquared[6:]) @ modalForces[6:]
    nodalDisplacements = waveHeight / 2 * dryVibrationModesNormalized[:, 6:] @ elasticModalDisplacements
    return beam.InternalForce(hullLength / 2, nodalDisplacements, 'mv')

bendingMomentsFromFroudeKrylovForces = np.zeros([omegas.size], dtype = complex)
bendingMomentsFromDiffractionForces = np.zeros([omegas.size], dtype = complex)
bendingMomentsFromStructuralMassForces = np.zeros([omegas.size], dtype = complex)
bendingMomentsFromAddedMassForces = np.zeros([omegas.size], dtype = complex)
bendingMomentsFromRadiationForces = np.zeros([omegas.size], dtype = complex)
bendingMomentsFromHydrostaticStiffnessForces = np.zeros([omegas.size], dtype = complex)

for i in range(omegas.size):
    bendingMomentsFromFroudeKrylovForces[i] = MidshipsBendingMomentFromModalForces(froudeKrylovModalForces[i, :])
    bendingMomentsFromDiffractionForces[i] = MidshipsBendingMomentFromModalForces(diffractionModalForces[i, :])
    bendingMomentsFromStructuralMassForces[i] = MidshipsBendingMomentFromModalForces(-structuralMassModalForces[i, :])
    bendingMomentsFromAddedMassForces[i] = MidshipsBendingMomentFromModalForces(-addedMassModalForces[i, :])
    bendingMomentsFromRadiationForces[i] = MidshipsBendingMomentFromModalForces(-radiationModalForces[i, :])
    bendingMomentsFromHydrostaticStiffnessForces[i] = MidshipsBendingMomentFromModalForces(-hydrostaticStiffnessModalForces[i, :])

# integration of excitation pressures to compute shear forces and bending moments
xVerticalLoads = np.linspace(-hullLength/2 + hullLength/panelsLength/2, hullLength/2 - hullLength/panelsLength/2, panelsLength)
verticalFroudeKrylovLoads = np.zeros_like(xVerticalLoads, dtype = froudeKrylovPressures.dtype)
verticalDiffractionLoads = np.zeros_like(xVerticalLoads, dtype = froudeKrylovPressures.dtype)

for i in range(hullMesh.nb_faces):
    xPositionIndex = np.argmin(np.abs(xVerticalLoads - hullMesh.faces_centers[i, 0]))

    verticalFroudeKrylovLoads[xPositionIndex] -= waveHeight/2 * froudeKrylovPressures[i] * hullMesh.faces_areas[i] * hullMesh.faces_normals[i, 2] / (hullLength/panelsLength)
    verticalDiffractionLoads[xPositionIndex] -= waveHeight/2 * diffractionResults.pressure[i] * hullMesh.faces_areas[i] * hullMesh.faces_normals[i, 2] / (hullLength/panelsLength)

verticalFroudeKrylovLoads -= np.sum(verticalFroudeKrylovLoads * xVerticalLoads) / np.sum(xVerticalLoads**2) * xVerticalLoads
verticalDiffractionLoads -= np.sum(verticalDiffractionLoads * xVerticalLoads) / np.sum(xVerticalLoads**2) * xVerticalLoads

verticalFroudeKrylovLoads -= np.mean(verticalFroudeKrylovLoads)
verticalDiffractionLoads -= np.mean(verticalDiffractionLoads)

xForceDistributions = np.linspace(-hullLength/2, hullLength/2, panelsLength + 1)
froudeKrylovShearForces = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)
diffractionShearForces = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)

for i in range(1, xForceDistributions.size):
    froudeKrylovShearForces[i] = froudeKrylovShearForces[i - 1] - verticalFroudeKrylovLoads[i - 1] * hullLength/panelsLength
    diffractionShearForces[i] = diffractionShearForces[i - 1] - verticalDiffractionLoads[i - 1] * hullLength/panelsLength

froudeKrylovBendingMoments = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)
diffractionBendingMoments = np.zeros_like(xForceDistributions, dtype = froudeKrylovPressures.dtype)

for i in range(1, xForceDistributions.size):
    froudeKrylovBendingMoments[i] = froudeKrylovBendingMoments[i - 1] + 0.5 * (froudeKrylovShearForces[i - 1] + froudeKrylovShearForces[i]) * hullLength/panelsLength
    diffractionBendingMoments[i] = diffractionBendingMoments[i - 1] + 0.5 * (diffractionShearForces[i - 1] + diffractionShearForces[i]) * hullLength/panelsLength

excitationShearForces = froudeKrylovShearForces + diffractionShearForces
excitationBendingMoments = froudeKrylovBendingMoments + diffractionBendingMoments



# OUTPUT

print()
print('Dry natural frequencies')
print('Number     Frequency (Hz)')
counter = 0
for i in range(dryNaturalFrequenciesHz.size):
    if counter == 0:
        if np.isnan(dryNaturalFrequenciesHz[i]):
            continue
        if dryNaturalFrequenciesHz[i] == 0:
            continue
    counter += 1
    if counter == 10:
        break
    print('%2d         %.2f'%(counter, dryNaturalFrequenciesHz[i]))
print()
print('Bending moment amplitudes for each wave frequency')
print('Omega [rad/s]      Bending moment [Nm]')
for i in range(omegas.size):
    print('%.2f               %.2f'%(omegas[i], midshipsBendingMomentAmplitudes[i]))
print()

# for i in range(6):
#     animation = hullBody.animate(motion = 'mode%d'%i, loop_duration = 1)
#     animation.run()


# paper results
import pandas as pd
rho = 1000 #Water density (kg/m^3)
g = 9.81 #Gravitational acceleration (m/s^2)
L = 1.52 #Model length (m)
B = 0.22 #Model breadth (m)

S2M = [132365.4187, 155869.0311, 156046.5947, 158543.3094, 154959.3894]

Strain2 = []

Strain2.append([0.023634916, 0.029264058, 0.029426426, 0.031062434, 0.031127915, 0.031826421, 0.031822112, 0.02310694])
Strain2.append([0.02380636, 0.029098951, 0.030430322, 0.031381076, 0.031749035, 0.03113457, 0.030265477, 0.022310203])
Strain2.append([0.024182254, 0.028697271, 0.02985464, 0.031329122, 0.031449131, 0.030472916, 0.031032499, 0.024203041])

MARS_responses = pd.read_pickle('validation/MARS_barge_response.pkl')

lL = [2.0, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.6]
Ll = [1/length for length in lL]
Ll_full = MARS_responses['L/lambda']

midshipsBendingMomentCoefsExperimental = [[S2M[2] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain2]
midshipsBendingMomentCoefs2DNumerical = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 1/2']]

if not os.path.exists('solutions/%dmodesProper'%numberModes):
    os.makedirs('solutions/%dmodesProper'%numberModes)

plt.figure()
plt.title('Midships bending moment coefficient for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs, 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in midshipsBendingMomentCoefsExperimental:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, midshipsBendingMomentCoefs2DNumerical, 'k--', label = '2D Hydroelascity')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/midshipsBendingMomentCoefs.png'%numberModes)

plt.figure()
plt.title('Heave RAO')
plt.plot(hullLength/wavelengths, heaveAmplitudes / (waveHeight / 2), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave RAO [m/m]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveRAO.png'%numberModes)

plt.figure()
plt.title('Pitch RAO')
plt.plot(hullLength/wavelengths, pitchAmplitudes / (waveHeight / wavelengths * np.pi), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch RAO [rad/rad]')
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchRAO.png'%numberModes)

plt.figure()
plt.title('Heave excitation force')
plt.plot(hullLength/wavelengths, np.abs(hydrodynamicResults.excitation_force.values[:, 0, 2]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave excitation force over square root of ship mass per unit wave amplitude [N/sqrt(kg) / m]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveExcitation.png'%numberModes)

plt.figure()
plt.title('Pitch excitation force')
plt.plot(hullLength/wavelengths, np.abs(hydrodynamicResults.excitation_force.values[:, 0, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch excitation force over square root of pitch inertia per unit wave amplitude [Nm/sqrt(kg m2) / m]')
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchExcitation.png'%numberModes)

plt.figure()
plt.title('Heave transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 2, 2]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveTransferFunction.png'%numberModes)

plt.figure()
plt.title('Pitch transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 4, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchTransferFunction.png'%numberModes)

plt.figure()
plt.title('Heave to pitch transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 2, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave to pitch transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveToPitchTransferFunction.png'%numberModes)

plt.figure()
plt.title('Vertical bending moment distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(x, np.imag(bendingMomentDistribution), 'g', label = 'Imaginary part, full springing results')
plt.plot(x, np.real(bendingMomentDistribution), 'k', label = 'Real part, full springing results')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationBendingMoments), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationBendingMoments), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.ylim([-1.2, 4.5])
plt.xlabel('x [m]')
plt.ylabel('Vertical bending moment [Nm]')
plt.legend()
plt.savefig('solutions/%dmodesProper/bendingMomentDistribution.png'%numberModes)

plt.figure()
plt.title('Vertical shear force distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(x, np.imag(shearForceDistribution), 'g', label = 'Imaginary part, full springing results')
plt.plot(x, np.real(shearForceDistribution), 'k', label = 'Real part, full springing results')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationShearForces), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationShearForces), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.ylim([-6, 11])
plt.xlabel('x [m]')
plt.ylabel('Vertical shear force [N]')
plt.legend()
plt.savefig('solutions/%dmodesProper/shearForceDistribution.png'%numberModes)

plt.figure(figsize = (20,10))
plt.title('Midships bending moment force decomposition')
plt.plot(omegas, np.real(midshipsBendingMoments), 'ko', label = 'Full, real')
plt.plot(omegas, np.imag(midshipsBendingMoments), 'kx', label = 'Full, imag')
plt.plot(omegas, np.real(bendingMomentsFromFroudeKrylovForces), 'yo', label = 'Froude-Krylov, real')
plt.plot(omegas, np.imag(bendingMomentsFromFroudeKrylovForces), 'yx', label = 'Froude-Krylov, imag')
plt.plot(omegas, np.real(bendingMomentsFromDiffractionForces), 'co', label = 'Diffraction, real')
plt.plot(omegas, np.imag(bendingMomentsFromDiffractionForces), 'cx', label = 'Diffraction, imag')
plt.plot(omegas, np.real(bendingMomentsFromStructuralMassForces), 'mo', label = 'Structural mass, real')
plt.plot(omegas, np.imag(bendingMomentsFromStructuralMassForces), 'mx', label = 'Structural mass, imag')
plt.plot(omegas, np.real(bendingMomentsFromAddedMassForces), 'ro', label = 'Added mass, real')
plt.plot(omegas, np.imag(bendingMomentsFromAddedMassForces), 'rx', label = 'Added mass, imag')
plt.plot(omegas, np.real(bendingMomentsFromRadiationForces), 'go', label = 'Radiation, real')
plt.plot(omegas, np.imag(bendingMomentsFromRadiationForces), 'gx', label = 'Radiation, imag')
plt.plot(omegas, np.real(bendingMomentsFromHydrostaticStiffnessForces), 'bo', label = 'Hydrostatic stiffness, real')
plt.plot(omegas, np.imag(bendingMomentsFromHydrostaticStiffnessForces), 'bx', label = 'Hydrostatic stiffness, imag')
plt.xlabel('omega [rad/s]')
plt.ylabel('Bending moment [Nm]')
plt.legend()
plt.savefig('solutions/%dmodesProper/bendingMomentDecomposition.png'%numberModes)

plt.show()


motion = {}

dofIndex = 0

for dof in hullBody.dofs.keys():
    motion[dof] = waveHeight / 2 * modalSpringingResults.modalAmplitudes.values[omegaIndex, 0, dofIndex]
    dofIndex += 1

animation = hullBody.animate(motion = motion, loop_duration = 1)
free_surface = cpt.FreeSurface(x_range=(-5, 5), y_range=(-5, 5), nx=150, ny=150)
animation.add_free_surface(free_surface, faces_elevation = waveHeight / 2 * airy_waves_free_surface_elevation(free_surface.mesh, diffractionProblem))
animation.run()