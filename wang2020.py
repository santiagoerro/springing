import numpy as np
import xarray as xr
import springing as spr
import capytaine as cpt
import sys
import matplotlib.pyplot as plt
import os
cpt.set_logging('INFO')



# INPUT

# water properties
waterDensity = 1000

# gravity
gravity = 9.81

# mass distribution, ship divided in 20 stations
linearMassDensityBetweenStations = np.array([50, 121, 119, 118, 78, 68, 71, 73, 83, 76, 80, 71, 73, 90, 70, 71, 76, 71, 38, 90]) * 1000 / 32**2

# hull properties
draft = 0.164625
lengthBetweenPerpendiculars = 130 / 32
hullBreadth = 25.6 / 32

# beam properties
beamSegments = 40

# material properties
youngsModulus = 2.1e11
shearModulus = youngsModulus / (2 * (1 + 0.29))

# section properties
sectionalAreas = np.ones([beamSegments]) * 1.2214e-3
verticalAreaMoments = np.ones([beamSegments]) * 5.009e-7
horizontalAreaMoments = np.ones([beamSegments]) * 4.0808e-6
verticalTimoshenkoCoefs = np.ones([beamSegments]) * 0.360
horizontalTimoshenkoCoefs = np.ones([beamSegments]) * 0.394
torsionConstants = np.ones([beamSegments]) * 1.341e-7
warpingConstants = np.ones([beamSegments]) * 1.650e-9
zNeutralAxis = -draft / 2
zTwistCenter = zNeutralAxis - 0.0376

# less stiff section properties
verticalAreaMomentsLessStiff = np.ones([beamSegments]) * 1.0018e-8
torsionConstantsLessStiff = np.ones([beamSegments]) * 6.705e-9
warpingConstantsLessStiff = np.ones([beamSegments]) * 8.250e-11

# wave characteristics
waveAmplitude = 1.2 / 32

shipLengthOverWavelength = np.linspace(0.05, 5, 60)
waveDirections = np.array([3*np.pi/4, np.pi])

# number of modal components considered
numberModes = 10



# CALCULATIONS
# base backbone

if not beamSegments % linearMassDensityBetweenStations.size == 0:
    sys.exit('Number of beam segments must be a multiple of 20')
elementsPerStation = beamSegments // linearMassDensityBetweenStations.size

linearDensitiesBeam = np.zeros([beamSegments])
for i in range(linearMassDensityBetweenStations.size):
    linearDensitiesBeam[elementsPerStation * i : elementsPerStation * (i + 1)] = linearMassDensityBetweenStations[i]

beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(0, lengthBetweenPerpendiculars, beamSegments + 1)
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
beamDefinition['linearDensities'] = linearDensitiesBeam
beamDefinition['zCentersOfMass'] = np.ones([beamSegments]) * zNeutralAxis
beamDefinition['rollInertias'] = linearDensitiesBeam * (hullBreadth*0.35)**2

beam = spr.Beam(beamDefinition)


stbdMesh = cpt.load_mesh('application/wang2020.med', file_format = 'salome')
mesh = cpt.ReflectionSymmetricMesh(stbdMesh, cpt.xOz_Plane)

dryNaturalFrequenciesSquared, dryModeShapes, modalDofs = beam.CalculateModalDOFs(mesh, beam.numberNodes * 7)
body = cpt.FloatingBody(mesh, modalDofs)


hydrostaticStiffness = spr.ComputeHydrostaticStiffness(body, waterDensity, gravity)

waterDepth = np.inf
waveWavenumbers = 2*np.pi * shipLengthOverWavelength / lengthBetweenPerpendiculars
omegas = np.sqrt(gravity * waveWavenumbers)

testMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(body.dofs)[:numberModes],
    'water_depth': waterDepth,
    'rho': waterDensity,
    'g': gravity
})

hydrodynamicResults = cpt.BEMSolver().fill_dataset(testMatrix, body)

modalSpringingResults = spr.ModalProperSpringingResults(dryNaturalFrequenciesSquared, hydrostaticStiffness, hydrodynamicResults)


heaveAmplitudes = np.abs(waveAmplitude * modalSpringingResults.modalAmplitudes.values[:, :, 2] * dryModeShapes[2, 2])
pitchAmplitudes = np.abs(waveAmplitude * modalSpringingResults.modalAmplitudes.values[:, :, 4] * dryModeShapes[5, 4])

bendingMoments = np.zeros([omegas.size, waveDirections.size], dtype = np.complex128)
torsionMoments = np.zeros([omegas.size, waveDirections.size], dtype = np.complex128)

for i in range(omegas.size):
    for j in range(waveDirections.size):
        nodalDisplacements = waveAmplitude * dryModeShapes @ modalSpringingResults.modalAmplitudes.values[i, j, :]
        bendingMoments[i, j] = beam.InternalForce(lengthBetweenPerpendiculars/2, nodalDisplacements, 'mv')
        torsionMoments[i, j] = beam.InternalForce(lengthBetweenPerpendiculars/2, nodalDisplacements, 't')

bendingMomentAmplitudes = np.abs(bendingMoments)
torsionMomentAmplitudes = np.abs(torsionMoments)

bendingMomentCoefs = bendingMomentAmplitudes / (waterDensity * gravity * lengthBetweenPerpendiculars**2 * hullBreadth * waveAmplitude)
torsionMomentCoefs = torsionMomentAmplitudes / (waterDensity * gravity * lengthBetweenPerpendiculars**2 * hullBreadth * waveAmplitude)


# less stiff backbone

beamDefinitionLessStiff = beamDefinition

beamDefinitionLessStiff['verticalAreaMoments'] = verticalAreaMomentsLessStiff
beamDefinitionLessStiff['torsionConstants'] = torsionConstantsLessStiff
beamDefinitionLessStiff['warpingConstants'] = warpingConstantsLessStiff

beamLessStiff = spr.Beam(beamDefinitionLessStiff)


dryNaturalFrequenciesSquaredLessStiff, dryModeShapesLessStiff, modalDofsLessStiff = beamLessStiff.CalculateModalDOFs(mesh, beamLessStiff.numberNodes * 7)
bodyLessStiff = cpt.FloatingBody(mesh, modalDofsLessStiff)


hydrostaticStiffnessLessStiff = spr.ComputeHydrostaticStiffness(bodyLessStiff, waterDensity, gravity)


testMatrixLessStiff = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(bodyLessStiff.dofs)[:numberModes],
    'water_depth': waterDepth,
    'rho': waterDensity,
    'g': gravity
})

hydrodynamicResultsLessStiff = cpt.BEMSolver().fill_dataset(testMatrixLessStiff, bodyLessStiff)

modalSpringingResultsLessStiff = spr.ModalProperSpringingResults(dryNaturalFrequenciesSquaredLessStiff, hydrostaticStiffnessLessStiff, hydrodynamicResultsLessStiff)


heaveAmplitudesLessStiff = np.abs(waveAmplitude * modalSpringingResultsLessStiff.modalAmplitudes.values[:, :, 2] * dryModeShapesLessStiff[2, 2])
pitchAmplitudesLessStiff = np.abs(waveAmplitude * modalSpringingResultsLessStiff.modalAmplitudes.values[:, :, 4] * dryModeShapesLessStiff[5, 4])

bendingMomentsLessStiff = np.zeros([omegas.size, waveDirections.size], dtype = np.complex128)
torsionMomentsLessStiff = np.zeros([omegas.size, waveDirections.size], dtype = np.complex128)

for i in range(omegas.size):
    for j in range(waveDirections.size):
        nodalDisplacements = waveAmplitude * dryModeShapesLessStiff @ modalSpringingResultsLessStiff.modalAmplitudes.values[i, j, :]
        bendingMomentsLessStiff[i, j] = beamLessStiff.InternalForce(lengthBetweenPerpendiculars/2, nodalDisplacements, 'mv')
        torsionMomentsLessStiff[i, j] = beamLessStiff.InternalForce(lengthBetweenPerpendiculars/2, nodalDisplacements, 't')

bendingMomentAmplitudesLessStiff = np.abs(bendingMomentsLessStiff)
torsionMomentAmplitudesLessStiff = np.abs(torsionMomentsLessStiff)

bendingMomentCoefsLessStiff = bendingMomentAmplitudesLessStiff / (waterDensity * gravity * lengthBetweenPerpendiculars**2 * hullBreadth * waveAmplitude)
torsionMomentCoefsLessStiff = torsionMomentAmplitudesLessStiff / (waterDensity * gravity * lengthBetweenPerpendiculars**2 * hullBreadth * waveAmplitude)



# OUTPUT

if not os.path.exists('application/plots'):
    os.makedirs('application/plots')

plt.figure()
plt.title('Heave RAO in head waves')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 1] / waveAmplitude, 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave RAO [m/m]')
plt.savefig('application/plots/heaveRAOHead.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Pitch RAO in head waves')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 1] / (waveAmplitude * waveWavenumbers), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch RAO [rad/rad]')
plt.savefig('application/plots/pitchRAOHead.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Midships bending moment coefficient for different head waves')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 1], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'$\mathregular{CM_b}$')
plt.savefig('application/plots/bendingMomentCoefsHead.png', dpi = 200, bbox_inches = 'tight')

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (9.7, 4.5))
fig.suptitle('RAOs in head waves for both backbones.')
axs[0].set_title('Heave RAO')
axs[0].plot(shipLengthOverWavelength, heaveAmplitudes[:, 1] / waveAmplitude, 'ko', label = 'Base backbone')
axs[0].plot(shipLengthOverWavelength, heaveAmplitudesLessStiff[:, 1] / waveAmplitude, 'bx', label = 'Less stiff backbone')
axs[0].set(xlabel = 'Ship length / wavelength', ylabel = 'Heave RAO [m/m]')
axs[0].legend()
axs[1].set_title('Pitch RAO')
axs[1].plot(shipLengthOverWavelength, pitchAmplitudes[:, 1] / (waveAmplitude * waveWavenumbers), 'ko', label = 'Base backbone')
axs[1].plot(shipLengthOverWavelength, pitchAmplitudesLessStiff[:, 1] / (waveAmplitude * waveWavenumbers), 'bx', label = 'Less stiff backbone')
axs[1].set(xlabel = 'Ship length / wavelength', ylabel = 'Pitch RAO [rad/rad]')
axs[1].legend()
fig.savefig('application/plots/headRAOs.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Midships bending moment coefficient for different head waves, both backbones')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 1], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, bendingMomentCoefsLessStiff[:, 1], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'$\mathregular{CM_b}$')
plt.legend()
plt.savefig('application/plots/bendingMomentCoefsHeadComparison.png', dpi = 200, bbox_inches = 'tight')


plt.figure()
plt.title('Heave modal transfer function')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 2, 2]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'Heave modal transfer function [$\mathregular{s^2}$]')
plt.savefig('application/plots/transferFunctionHeave.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Pitch modal transfer function')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 4, 4]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'Pitch modal transfer function [$\mathregular{s^2}$]')
plt.savefig('application/plots/transferFunctionPitch.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Roll modal transfer function')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 3, 3]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'Roll modal transfer function [$\mathregular{s^2}$]')
plt.savefig('application/plots/transferFunctionRoll.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Two-node vertical bending modal transfer function')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 6, 6]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'V2 transfer function [$\mathregular{s^2}$]')
plt.savefig('application/plots/transferFunctionV2.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('H1T1 modal transfer function')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 7, 7]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'H1T1 transfer function [$\mathregular{s^2}$]')
plt.savefig('application/plots/transferFunctionH1T1.png', dpi = 200, bbox_inches = 'tight')


plt.figure()
plt.title('Heave RAO in 45 degree bow waves')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 0] / waveAmplitude, 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave RAO [m/m]')
plt.savefig('application/plots/heaveRAOBow.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Pitch RAO in 45 degree bow waves')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 0] / (waveAmplitude * waveWavenumbers), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch RAO [rad/rad]')
plt.savefig('application/plots/pitchRAOBow.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Midships bending moment coefficient for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'$\mathregular{CM_b}$')
plt.savefig('application/plots/bendingMomentCoefsBow.png', dpi = 200, bbox_inches = 'tight')

plt.figure()
plt.title('Midships torsional moment coefficient for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, torsionMomentCoefs[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel(r'$\mathregular{CM_t}$')
plt.savefig('application/plots/torsionMomentCoefsBow.png', dpi = 200, bbox_inches = 'tight')

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (9.7, 4.5))
fig.suptitle('RAOs in 45 degree bow waves for both backbones.')
axs[0].set_title('Heave RAO')
axs[0].plot(shipLengthOverWavelength, heaveAmplitudes[:, 0] / waveAmplitude, 'ko', label = 'Base backbone')
axs[0].plot(shipLengthOverWavelength, heaveAmplitudesLessStiff[:, 0] / waveAmplitude, 'bx', label = 'Less stiff backbone')
axs[0].set(xlabel = 'Ship length / wavelength', ylabel = 'Heave RAO [m/m]')
axs[0].legend()
axs[1].set_title('Pitch RAO')
axs[1].plot(shipLengthOverWavelength, pitchAmplitudes[:, 0] / (waveAmplitude * waveWavenumbers), 'ko', label = 'Base backbone')
axs[1].plot(shipLengthOverWavelength, pitchAmplitudesLessStiff[:, 0] / (waveAmplitude * waveWavenumbers), 'bx', label = 'Less stiff backbone')
axs[1].set(xlabel = 'Ship length / wavelength', ylabel = 'Pitch RAO [rad/rad]')
axs[1].legend()
fig.savefig('application/plots/bowRAOs.png', dpi = 200, bbox_inches = 'tight')

fig, axs = plt.subplots(1, 2, layout = 'constrained', figsize = (9.7, 4.5))
fig.suptitle('Midships bending and torsional moment coefficients for different 45 degree bow waves, both backbones.')
axs[0].set_title('Bending moment coefficient')
axs[0].plot(shipLengthOverWavelength, bendingMomentCoefs[:, 0], 'ko', label = 'Base backbone')
axs[0].plot(shipLengthOverWavelength, bendingMomentCoefsLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
axs[0].set(xlabel = 'Ship length / wavelength', ylabel = r'$\mathregular{CM_b}$')
axs[0].legend()
axs[1].set_title('Torsional moment coefficient')
axs[1].plot(shipLengthOverWavelength, torsionMomentCoefs[:, 0], 'ko', label = 'Base backbone')
axs[1].plot(shipLengthOverWavelength, torsionMomentCoefsLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
axs[1].set(xlabel = 'Ship length / wavelength', ylabel = r'$\mathregular{CM_t}$')
axs[1].legend()
fig.savefig('application/plots/bowMomentCoefs.png', dpi = 200, bbox_inches = 'tight')

plt.show()