import numpy as np
import xarray as xr
import springing as spr
import capytaine as cpt
import sys
import matplotlib.pyplot as plt
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

shipLengthOverWavelength = np.linspace(0.5, 5, 60)
# shipLengthOverWavelength = np.linspace(4.538, 4.542, 30)
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



# OUTPUT

plt.figure()
plt.title('Heave amplitudes for different head waves')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 1], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave amplitude [m]')

plt.figure()
plt.title('Pitch amplitudes for different head waves')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 1], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch amplitude [rad]')

plt.figure()
plt.title('Midships bending moment coefficient for different head waves')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 1], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')

plt.figure()
plt.title('Heave amplitudes for different head waves, both backbones')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 1], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, heaveAmplitudesLessStiff[:, 1], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave amplitude [m]')
plt.legend()

plt.figure()
plt.title('Pitch amplitudes for different head waves, both backbones')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 1], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, pitchAmplitudesLessStiff[:, 1], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch amplitude [rad]')
plt.legend()

plt.figure()
plt.title('Midships bending moment coefficient for different head waves, both backbones')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 1], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, bendingMomentCoefsLessStiff[:, 1], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()

plt.figure()
plt.title('Transfer function from first flexible mode to itself, modal force to amplitude')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 6, 6]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('X7/F7 [1/s2]')

plt.figure()
plt.title('Transfer function from second flexible mode to itself, modal force to amplitude')
plt.plot(shipLengthOverWavelength, np.abs(modalSpringingResultsLessStiff.modalAmplitudesFromForcesMatrices[:, 7, 7]), 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('X8/F8 [1/s2]')


plt.figure()
plt.title('Heave amplitudes for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave amplitude [m]')

plt.figure()
plt.title('Pitch amplitudes for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch amplitude [rad]')

plt.figure()
plt.title('Midships bending moment coefficient for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')

plt.figure()
plt.title('Midships torsion moment amplitudes for different 45 degree bow waves')
plt.plot(shipLengthOverWavelength, torsionMomentAmplitudes[:, 0], 'ko')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Mt [Nm]')

plt.figure()
plt.title('Heave amplitudes for different 45 degree bow waves, both backbones')
plt.plot(shipLengthOverWavelength, heaveAmplitudes[:, 0], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, heaveAmplitudesLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Heave amplitude [m]')
plt.legend()

plt.figure()
plt.title('Pitch amplitudes for different 45 degree bow waves, both backbones')
plt.plot(shipLengthOverWavelength, pitchAmplitudes[:, 0], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, pitchAmplitudesLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Pitch amplitude [rad]')
plt.legend()

plt.figure()
plt.title('Midships bending moment coefficient for different 45 degree bow waves, both backbones')
plt.plot(shipLengthOverWavelength, bendingMomentCoefs[:, 0], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, bendingMomentCoefsLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()

plt.figure()
plt.title('Midships torsion moment amplitudes for different 45 degree bow waves, both backbones')
plt.plot(shipLengthOverWavelength, torsionMomentAmplitudes[:, 0], 'ko', label = 'Base backbone')
plt.plot(shipLengthOverWavelength, torsionMomentAmplitudesLessStiff[:, 0], 'bx', label = 'Less stiff backbone')
plt.xlabel('Ship length / wavelength')
plt.ylabel('Mt [Nm]')
plt.legend()

plt.show()