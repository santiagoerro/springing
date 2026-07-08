import numpy as np
import xarray as xr
import springing as spr
import capytaine as cpt
import sys



# INPUT

# water properties
waterDensity = 1000

# gravity
gravity = 9.81

# mass distribution, ship divided in 20 stations
linearMassDensityTonPerMeterBetweenStations = np.array([50, 121, 119, 118, 78, 68, 71, 73, 83, 76, 80, 71, 73, 90, 70, 71, 76, 71, 38, 90]) * 1000 / 32**2

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
verticalAreaMoments = np.ones([beamSegments]) * 5.009e-7
horizontalAreaMoments = np.ones([beamSegments]) * 4.0808e-6
sectionalAreas = np.ones([beamSegments]) * 1.2214e-3
verticalTimoshenkoCoefs = np.ones([beamSegments]) * 5/6
horizontalTimoshenkoCoefs = np.ones([beamSegments]) * 5/6
torsionConstants = np.ones([beamSegments]) * 1.341e-7
warpingConstants = np.ones([beamSegments]) * shearModulus * 1.341e-7 / (youngsModulus * 20**2) * lengthBetweenPerpendiculars**2
zNeutralAxis = -draft / 2
zTwistCenter = zNeutralAxis - 0.0032

# wave characteristics
waveAmplitude = 1.2 / 32

wavelengthOverShipLength = np.array([0.9, 1.1])
waveDirection = np.pi

# boat speeds
froudeNumbers = np.array([0.144, 0.202])

# number of modal components considered
numberModes = 10



# CALCULATIONS

if not beamSegments % linearMassDensityTonPerMeterBetweenStations.size == 0:
    sys.exit('Number of beam segments must be a multiple of 20')
elementsPerStation = beamSegments // linearMassDensityTonPerMeterBetweenStations.size

linearDensitiesBeam = np.zeros([beamSegments])
for i in range(linearMassDensityTonPerMeterBetweenStations.size):
    linearDensitiesBeam[elementsPerStation * i : elementsPerStation * (i + 1)] = linearMassDensityTonPerMeterBetweenStations[i]

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


stbdMesh = cpt.load_mesh('validation/wang2020.med', file_format = 'salome')
mesh = cpt.ReflectionSymmetricMesh(stbdMesh, cpt.xOz_Plane)

dryNaturalFrequenciesSquared, dryModeShapes, modalDofs = beam.CalculateModalDOFs(mesh, beam.numberNodes * 7)
body = cpt.FloatingBody(mesh, modalDofs)


waterDepth = np.inf
waveWavenumbers = 2*np.pi / (wavelengthOverShipLength * lengthBetweenPerpendiculars)
omegas = np.sqrt(gravity * waveWavenumbers)
speeds = np.sqrt(gravity * lengthBetweenPerpendiculars) * froudeNumbers

encounterOmegas = omegas - np.cos(waveDirection) * speeds * waveWavenumbers


hydrostaticStiffness = spr.ComputeHydrostaticStiffness(body, waterDensity, gravity)


radiationTestMatrix = xr.Dataset(coords={
    'omega': encounterOmegas,
    'radiating_dof': list(body.dofs)[:numberModes],
    'water_depth': waterDepth,
    'rho': waterDensity,
    'g': gravity
})

radiationResults = cpt.BEMSolver().fill_dataset(radiationTestMatrix, body)

diffractionTestMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirection,
    'water_depth': waterDepth,
    'rho': waterDensity,
    'g': gravity
})

diffractionResults = cpt.BEMSolver().fill_dataset(diffractionTestMatrix, body)


modalAmplitudes = np.zeros([beam.numberNodes * 7, omegas.size])
for i in range(omegas.size):
    addedMass = np.zeros([beam.numberNodes * 7, beam.numberNodes * 7])
    addedMass[:numberModes, :] = radiationResults.added_mass.values[i, :, :]
    addedMass[numberModes:, :numberModes] = radiationResults.added_mass.values[i, :, numberModes:].transpose()
    addedMass = addedMass.transpose()

    radiationDamping = np.zeros([beam.numberNodes * 7, beam.numberNodes * 7])
    radiationDamping[:numberModes, :] = radiationResults.radiation_damping.values[i, :, :]
    radiationDamping[numberModes:, :numberModes] = radiationResults.radiation_damping.values[i, :, numberModes:].transpose()
    radiationDamping = radiationDamping.transpose()

    modalDisplacementsToForces = -encounterOmegas[i]**2 * (np.eye(beam.numberNodes * 7) + addedMass) - complex(0,1) * encounterOmegas[i] * radiationDamping + (np.diag(dryNaturalFrequenciesSquared) + hydrostaticStiffness.values)

    modalAmplitudes[:, i] = np.linalg.solve(modalDisplacementsToForces, diffractionResults.excitation_force.values[i, 0, :])


midshipsVerticalBendingMomentAmplitudeFirstCase = np.abs(beam.InternalForce(lengthBetweenPerpendiculars/2, waveAmplitude * dryModeShapes @ modalAmplitudes[:, 0], 'mv'))
midshipsVerticalBendingMomentAmplitudeSecondCase = np.abs(beam.InternalForce(lengthBetweenPerpendiculars/2, waveAmplitude * dryModeShapes @ modalAmplitudes[:, 1], 'mv'))

print('a')