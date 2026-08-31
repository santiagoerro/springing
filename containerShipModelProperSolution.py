import numpy as np
import capytaine as cpt
import xarray as xr
import os
import springing as spr
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation

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

# bending moment positions
bendingMomentPositions = np.array([3/4, 5/8, 1/2, 3/8, 1/4]) * hullLength

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
bendingMoments = np.zeros([bendingMomentPositions.size, omegas.size], dtype = np.complex128)
bendingMomentAmplitudes = np.zeros([bendingMomentPositions.size, omegas.size])
heaveAmplitudes = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes.values[:, 0, 2] * dryVibrationModesNormalized[2, 2])
pitchAmplitudes = np.abs(waveHeight / 2 * modalSpringingResults.modalAmplitudes.values[:, 0, 4] * dryVibrationModesNormalized[5, 4])

for i in range(omegas.size):
    displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[i, 0, :]
    for j in range(bendingMomentPositions.size):
        bendingMoments[j, i] = beam.InternalForce(bendingMomentPositions[j], displacements, 'mv')
        bendingMomentAmplitudes[j, i] = np.abs(bendingMoments[j, i])

bendingMomentCoefs = bendingMomentAmplitudes / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)

# wavelengths
wavelengths = hydrodynamicResults.wavelength.values

# bending moment and shear force distributions
x = np.linspace(0, hullLength, 500)
displacements = waveHeight / 2 * dryVibrationModesNormalized @ modalSpringingResults.modalAmplitudes.values[omegaIndex, 0, :]
bendingMomentDistribution = beam.InternalForce(x, displacements, 'mv')
shearForceDistribution = beam.InternalForce(x, displacements, 'sv')
np.save('data/6modeProperDisplacements.npy', displacements)

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
print('Midships bending moment amplitudes for each wave frequency')
print('Omega [rad/s]      Bending moment [Nm]')
for i in range(omegas.size):
    print('%.2f               %.2f'%(omegas[i], bendingMomentAmplitudes[2, i]))
print()


# paper results
import pandas as pd

MARS_responses = pd.read_pickle('validation/MARS_barge_response.pkl')

lL = [2.0, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.6]
Ll = [1/length for length in lL]
Ll_full = MARS_responses['L/lambda']

Heave1 = [0.73751406007521, 0.587961303166672, 0.396723470798733, 0.551115200253913, 0.552640373899697, 0.473923035341315, 0.30497318797963, 0.268711005147703]
Heave2 = [0.736804596255842, 0.600541397795775, 0.435720340002403, 0.561840672690008, 0.577242791966294, 0.454999905962747, 0.272375323274557, 0.294066049072504]
Heave3 = [0.745546266733696, 0.583323005191608, 0.446280462246718, 0.554438681464455, 0.556251221310476, 0.455865474510364, 0.303226852288591, 0.269567881495991]

Heave1 = [h1 ** 2 for h1 in Heave1]
Heave2 = [h2 ** 2 for h2 in Heave2]
Heave3 = [h3 ** 2 for h3 in Heave3]

HeaveMARS = MARS_responses['Heave']

Pitch1 = [8.72643201531739, 9.27223595661728, 9.38238178270851, 9.22259695073318, 8.8954977716883, 7.6900782818477, 6.66559572975357, 4.12941589153614]
Pitch2 = [8.74640514745736, 9.26039346917159, 9.54738151621977, 9.3351725914385, 9.07521036108446, 7.60072647902592, 6.49105824911448, 4.57893847336666]
Pitch3 = [8.79440800024845, 9.09976583481649, 9.45256463952212, 9.30061322085794, 8.79220999851533, 7.49774004759873, 6.83027046480925, 2.55617968532918]

Pitch1 = [np.pi/180 * (p1 ** 2) for p1 in Pitch1]
Pitch2 = [np.pi/180 * (p2 ** 2) for p2 in Pitch2]
Pitch3 = [np.pi/180 * (p3 ** 2) for p3 in Pitch3]

PitchMARS = MARS_responses['Pitch']

Strain0 = []
Strain1 = []
Strain2 = []
Strain3 = []
Strain4 = []

Strain0.append([0.018124984, 0.022618695, 0.023298096, 0.025116965, 0.025454998, 0.026175161, 0.027164676, 0.018977597])
Strain0.append([0.018263503, 0.022437595, 0.024109942, 0.02541972, 0.026036184, 0.025498779, 0.025682302, 0.018454928])
Strain0.append([0.018592155, 0.022084523, 0.023524231, 0.025229337, 0.025775674, 0.025250121, 0.026233943, 0.019836465])
Strain1.append([0.023851372, 0.029433457, 0.029601587, 0.031924556, 0.032265631, 0.03324423, 0.033919525, 0.023988808])
Strain1.append([0.024038963, 0.029248741, 0.030718343, 0.032348025, 0.032937128, 0.032481886, 0.032092476, 0.023165277])
Strain1.append([0.024509788, 0.02878194, 0.029925292, 0.032084797, 0.032550957, 0.031788558, 0.032777625, 0.025095487])
Strain2.append([0.023634916, 0.029264058, 0.029426426, 0.031062434, 0.031127915, 0.031826421, 0.031822112, 0.02310694])
Strain2.append([0.02380636, 0.029098951, 0.030430322, 0.031381076, 0.031749035, 0.03113457, 0.030265477, 0.022310203])
Strain2.append([0.024182254, 0.028697271, 0.02985464, 0.031329122, 0.031449131, 0.030472916, 0.031032499, 0.024203041])
Strain3.append([0.022485032, 0.027804172, 0.028050898, 0.029009122, 0.028996836, 0.028811349, 0.028682339, 0.02069366])
Strain3.append([0.022604209, 0.027723254, 0.028812319, 0.029350391, 0.029408922, 0.028358236, 0.027268316, 0.020188885])
Strain3.append([0.022888986, 0.027293777, 0.028492544, 0.02922189, 0.029127223, 0.027660774, 0.02799229, 0.022256689])
Strain4.append([0.01801357, 0.022403417, 0.022683151, 0.023219541, 0.023157699, 0.022515673, 0.022042877, 0.016059296])
Strain4.append([0.018081537, 0.022335745, 0.023220867, 0.023584137, 0.023422664, 0.022317655, 0.02104502, 0.015756252])
Strain4.append([0.01825508, 0.021962241, 0.02306033, 0.023439243, 0.023186253, 0.021637611, 0.02150482, 0.017565068])

rho = 1000 #Water density (kg/m^3)
g = 9.81 #Gravitational acceleration (m/s^2)
L = 1.52 #Model length (m)
B = 0.22 #Model breadth (m)

S2M = [132365.4187, 155869.0311, 156046.5947, 158543.3094, 154959.3894]
#S2M = [2.13E09 * 6.10E-06 / (0.13 - 0.0454075)] * 5

momentCoef0Exp = [[S2M[0] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain0]
momentCoef0MARS = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 3/4']]

momentCoef1Exp = [[S2M[1] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain1]
momentCoef1MARS = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 5/8']]

momentCoef2Exp = [[S2M[2] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain2]
momentCoef2MARS = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 1/2']]

momentCoef3Exp = [[S2M[3] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain3]
momentCoef3MARS = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 3/8']]

momentCoef4Exp = [[S2M[4] * strain ** 2 / (rho * g * L ** 2 * B) for strain in Strain_Run] for Strain_Run in Strain4]
momentCoef4MARS = [moment / (rho * g * L ** 2 * B) for moment in MARS_responses['BM 1/4']]

Stations = [5, 7.5, 10, 12.5, 15]
Stations_SB = [5, 10, 12.5]

z = [pos - 0.0454075 for pos in [0.13, 0.13, 0.028, 0.09, 0.028, 0.028]]

StrainDist = []
StrainDist.append([1.64436E-05, 2.64199E-05, 2.45896E-05, 2.13379E-05, 1.36095E-05, 1.70991E-05, 2.56243E-05, 2.2663E-05, 2.97811E-06, 1.58651E-06, 2.01319E-06, 4.05302E-06, 2.29376E-06, 1.23113E-05, 3.2125E-06, 4.19093E-06])
StrainDist.append([1.69688E-05, 2.71562E-05, 2.52324E-05, 2.16499E-05, 1.37332E-05, 1.76751E-05, 2.61086E-05, 2.3153E-05, 3.04294E-06, 1.61914E-06, 2.08546E-06, 4.12609E-06, 2.36466E-06, 1.2597E-05, 3.42853E-06, 3.95166E-06])
StrainDist.append([1.70238E-05, 2.71497E-05, 2.53428E-05, 2.17388E-05, 1.37752E-05, 1.76901E-05, 2.61392E-05, 2.33025E-05, 3.0509E-06, 1.57954E-06, 2.10893E-06, 4.15847E-06, 2.3709E-06, 1.26582E-05, 3.49951E-06, 3.88028E-06])
StrainDist = [[item/max(line) for item in line] for line in StrainDist]


aqwaRAOs = pd.read_csv('validation/aqwaRAOs.csv')

aqwaOmegas = np.array(aqwaRAOs['Wave Frequency (Hz)']) * 2*np.pi
aqwaMidshipsBendingMomentAmplitudes = np.array(aqwaRAOs['Line A (N.m/m)']) * waveHeight/2
aqwaMidshipsBendingMomentsReal = np.array(aqwaRAOs['Line B (N.m/m)']) * waveHeight/2
aqwaMidshipsBendingMomentsImaginary = np.array(aqwaRAOs['Line C (N.m/m)']) * waveHeight/2

def EquationsAquaWavenumbers(wavenumbers: np.ndarray):
    return gravity * wavenumbers * np.tanh(wavenumbers * waterDepths[0]) - aqwaOmegas**2

aqwaWavenumbers = fsolve(EquationsAquaWavenumbers, aqwaOmegas**2 / gravity)
aqwaWavelengths = 2*np.pi / aqwaWavenumbers

midshipsBendingMomentCoefsAqwa = aqwaMidshipsBendingMomentAmplitudes / (rho * g * L ** 2 * B * waveHeight/2)

heaveAmplitudesAqwa = np.array(aqwaRAOs['Line G (m/m)']) * waveHeight/2
pitchAmplitudesAqwa = np.pi/180 * np.array(aqwaRAOs['Line I (°/m)']) * waveHeight/2

aqwaForceDistributions = pd.read_csv('validation/aqwaForceDistributions.csv')

xAqwa = np.array(aqwaForceDistributions['Position (m)'])
shearForceAqwaReal = np.flip(np.array(aqwaForceDistributions['Line E (N/m)']) * waveHeight/2)
shearForceAqwaImaginary = np.flip(np.array(aqwaForceDistributions['Line F (N/m)']) * waveHeight/2)
bendingMomentAqwaReal = np.flip(np.array(aqwaForceDistributions['Line B (N.m/m)']) * waveHeight/2)
bendingMomentAqwaImaginary = np.flip(np.array(aqwaForceDistributions['Line C (N.m/m)']) * waveHeight/2)

if not os.path.exists('solutions/%dmodesProper'%numberModes):
    os.makedirs('solutions/%dmodesProper'%numberModes)

plt.figure()
plt.title('Bending moment coefficient at x = 3/4 L for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs[0, :], 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in momentCoef0Exp:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, momentCoef0MARS, 'k--', label = '2D Hydroelascity')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
# plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/0-75LBendingMomentCoefs.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Bending moment coefficient at x = 5/8 L for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs[1, :], 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in momentCoef1Exp:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, momentCoef1MARS, 'k--', label = '2D Hydroelascity')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
# plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/0-625LBendingMomentCoefs.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Midships bending moment coefficient for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs[2, :], 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in momentCoef2Exp:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, momentCoef2MARS, 'k--', label = '2D Hydroelascity')
plt.plot(hullLength/aqwaWavelengths, midshipsBendingMomentCoefsAqwa, 'k:', label = 'Ansys Aqwa')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
# plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/midshipsBendingMomentCoefs.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Bending moment coefficient at x = 3/8 L for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs[3, :], 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in momentCoef3Exp:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, momentCoef3MARS, 'k--', label = '2D Hydroelascity')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
# plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/0-375LBendingMomentCoefs.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Bending moment coefficient at x = 1/4 L for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs[4, :], 'ko', label = 'Capytaine Hydroelasticity')
flag = True
for series in momentCoef4Exp:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, momentCoef4MARS, 'k--', label = '2D Hydroelascity')
plt.xlim([0, hullLength/wavelengths[-1] * 1.03])
# plt.ylim([0, 0.05])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('solutions/%dmodesProper/0-25LBendingMomentCoefs.png'%numberModes, dpi = 200)

counter = 0
plt.figure()
plt.title('Heave RAO')
plt.plot(hullLength/wavelengths, heaveAmplitudes / (waveHeight / 2), 'ko', label = 'Capytaine Hydroelasticity')
for item in [Heave1, Heave2, Heave3]:
    if counter == 0:
        plt.plot(Ll, item, 'bo', label='Experiment')
        counter += 1
    else:
        plt.plot(Ll, item, 'bo')
if len(HeaveMARS) != 'none':
    plt.plot(Ll_full, HeaveMARS, 'k--', label='2D Hydroelasticity')
plt.plot(hullLength/aqwaWavelengths, heaveAmplitudesAqwa / (waveHeight / 2), 'k:', label = 'Ansys Aqwa')
plt.xlabel('Ship length / Wave length')
plt.ylabel('Heave RAO [m/m]')
plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveRAO.png'%numberModes, dpi = 200, bbox_inches = 'tight')

counter = 0
plt.figure()
plt.title('Pitch RAO')
for item in [Pitch1, Pitch2, Pitch3]:
    if counter == 0:
        plt.plot(Ll, item, 'bo', label='Experiment')
        counter += 1
    else:
        plt.plot(Ll, item, 'bo')
if len(PitchMARS) != 'none':
    plt.plot(Ll_full, PitchMARS, 'k--', label='2D Hydroelasticity')
plt.plot(hullLength/wavelengths, pitchAmplitudes / (waveHeight / 2), 'ko', label = 'Capytaine Hydroelasticity')
plt.plot(hullLength/aqwaWavelengths, pitchAmplitudesAqwa / (waveHeight / 2), 'k:', label = 'Ansys Aqwa')
plt.xlabel('Ship length / Wave length')
plt.ylabel('Pitch RAO [rad/m]')
plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchRAORadPerM.png'%numberModes, dpi = 200)

counter = 0
plt.figure()
plt.title('Pitch RAO')
plt.plot(hullLength/wavelengths, pitchAmplitudes / (waveHeight / wavelengths * np.pi), 'ko', label = 'Capytaine Hydroelasticity')
for item in [Pitch1, Pitch2, Pitch3]:
    if counter == 0:
        plt.plot(Ll, np.array(item) * hullLength / (2 * np.pi * np.array(Ll)), 'bo', label='Experiment')
        counter += 1
    else:
        plt.plot(Ll, np.array(item) * hullLength / (2 * np.pi * np.array(Ll)), 'bo')
if len(PitchMARS) != 'none':
    plt.plot(Ll_full, PitchMARS * hullLength / (2 * np.pi * Ll_full), 'k--', label='2D Hydroelasticity')
plt.plot(hullLength/aqwaWavelengths, pitchAmplitudesAqwa / (waveHeight / aqwaWavelengths * np.pi), 'k:', label = 'Ansys Aqwa')
plt.xlabel('Ship length / Wave length')
plt.ylabel('Pitch RAO [rad/rad]')
plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0, 0))
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchRAORadPerRad.png'%numberModes, dpi = 200, bbox_inches = 'tight')

# plt.figure()
# plt.title('Pitch RAO')
# # plt.plot(hullLength/wavelengths, pitchAmplitudes / (waveHeight / wavelengths * np.pi), 'ko', label = 'Capytaine Hydroelasticity')
# plt.plot(hullLength/wavelengths, pitchAmplitudes / (waveHeight / 2), 'ko', label = 'Capytaine Hydroelasticity')
# # plt.plot(hullLength/aqwaWavelengths, pitchAmplitudesAqwa / (waveHeight / aqwaWavelengths * np.pi), 'k:', label = 'Ansys Aqwa')
# plt.plot(hullLength/aqwaWavelengths, pitchAmplitudesAqwa / (waveHeight / 2), 'k:', label = 'Ansys Aqwa')
# plt.xlabel('Ship length / wavelength')
# # plt.ylabel('Pitch RAO [rad/rad]')
# plt.ylabel('Pitch RAO [rad/m]')
# plt.legend()
# plt.savefig('solutions/%dmodesProper/pitchRAO.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Heave excitation force')
plt.plot(hullLength/wavelengths, np.abs(hydrodynamicResults.excitation_force.values[:, 0, 2]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave excitation force over square root of ship mass per unit wave amplitude [N/sqrt(kg) / m]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveExcitation.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Pitch excitation force')
plt.plot(hullLength/wavelengths, np.abs(hydrodynamicResults.excitation_force.values[:, 0, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch excitation force over square root of pitch inertia per unit wave amplitude [Nm/sqrt(kg m2) / m]')
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchExcitation.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Heave transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 2, 2]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveTransferFunction.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Pitch transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 4, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/pitchTransferFunction.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Heave to pitch transfer function')
plt.plot(hullLength/wavelengths, np.abs(modalSpringingResults.modalAmplitudesFromForcesMatrices.values[:, 2, 4]), 'ko', label = 'Capytaine Hydroelasticity')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave to pitch transfer function [sqrt(kg)m / (sqrt(kg)m / s2)]')
plt.legend()
plt.savefig('solutions/%dmodesProper/heaveToPitchTransferFunction.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Vertical bending moment distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(x, np.imag(bendingMomentDistribution), 'g', label = 'Imaginary part, full springing results')
plt.plot(x, np.real(bendingMomentDistribution), 'k', label = 'Real part, full springing results')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationBendingMoments), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationBendingMoments), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.plot(xAqwa, bendingMomentAqwaImaginary, 'g:', label = 'Imaginary part, Ansys Aqwa results')
plt.plot(xAqwa, bendingMomentAqwaReal, 'k:', label = 'Real part, Ansys Aqwa results')
plt.ylim([-1.2, 4.5])
plt.xlabel('x [m]')
plt.ylabel('Vertical bending moment [Nm]')
plt.legend()
plt.savefig('solutions/%dmodesProper/bendingMomentDistribution.png'%numberModes, dpi = 200)

plt.figure()
plt.title('Vertical shear force distribution for omega = %.2f rad/s'%omegas[omegaIndex])
plt.plot(x, np.imag(shearForceDistribution), 'g', label = 'Imaginary part, full springing results')
plt.plot(x, np.real(shearForceDistribution), 'k', label = 'Real part, full springing results')
plt.plot(xForceDistributions + hullLength/2, np.imag(excitationShearForces), 'g--', label = 'Imaginary part, manual integration of excitation pressures')
plt.plot(xForceDistributions + hullLength/2, np.real(excitationShearForces), 'k--', label = 'Real part, manual integration of excitation pressures')
plt.plot(xAqwa, shearForceAqwaImaginary, 'g:', label = 'Imaginary part, Ansys Aqwa results')
plt.plot(xAqwa, shearForceAqwaReal, 'k:', label = 'Real part, Ansys Aqwa results')
plt.ylim([-6, 11])
plt.xlabel('x [m]')
plt.ylabel('Vertical shear force [N]')
plt.legend()
plt.savefig('solutions/%dmodesProper/shearForceDistribution.png'%numberModes, dpi = 200)

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (9, 5))
fig.suptitle('Comparison of vertical shear force and bending moment distributions for omega = 6.07 rad/s.')
axs[0].plot(x, np.imag(shearForceDistribution), 'g')
axs[0].plot(x, np.real(shearForceDistribution), 'k')
axs[0].plot(xForceDistributions + hullLength/2, np.imag(excitationShearForces), 'g--')
axs[0].plot(xForceDistributions + hullLength/2, np.real(excitationShearForces), 'k--')
axs[0].plot(xAqwa, shearForceAqwaImaginary, 'g:')
axs[0].plot(xAqwa, shearForceAqwaReal, 'k:')
axs[0].set(xlabel = 'x [m]', ylabel = 'Vertical shear force [N]')
axs[1].plot(x, np.imag(bendingMomentDistribution), 'g', label = 'Full springing results, imaginary part')
axs[1].plot(x, np.real(bendingMomentDistribution), 'k', label = 'Full springing results, real part')
axs[1].plot(xForceDistributions + hullLength/2, np.imag(excitationBendingMoments), 'g--', label = 'Integration of Capytaine excitation pressures, imaginary part')
axs[1].plot(xForceDistributions + hullLength/2, np.real(excitationBendingMoments), 'k--', label = 'Integration of Capytaine excitation pressures, real part')
axs[1].plot(xAqwa, bendingMomentAqwaImaginary, 'g:', label = 'Ansys Aqwa results, imaginary part')
axs[1].plot(xAqwa, bendingMomentAqwaReal, 'k:', label = 'Ansys Aqwa results, real part')
axs[1].set(xlabel = 'x [m]', ylabel = 'Vertical bending moment [Nm]')
fig.legend(loc = 'outside lower right')
fig.savefig('solutions/%dmodesProper/shearForceBendingMomentDistributions.png'%numberModes, dpi = 200, bbox_inches = 'tight')

plt.figure(figsize = (20,10))
plt.title('Midships bending moment force decomposition')
plt.plot(omegas, np.real(bendingMoments[2, :]), 'ko', label = 'Full, real')
plt.plot(omegas, np.imag(bendingMoments[2, :]), 'kx', label = 'Full, imag')
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
plt.savefig('solutions/%dmodesProper/bendingMomentDecomposition.png'%numberModes, dpi = 200)

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