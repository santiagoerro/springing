import numpy as np
import capytaine as cpt
import xarray as xr
import sys
from scipy.linalg import eigh
from capytaineBeamDeformingMesh import *
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
pointMasses = np.array([0.5, 0.5, 0.5, 1, 2, 1, 1])
pointMassStations = np.array([1, 2.5, 4.5, 6.5, 8, 15.5, 17.5]) # ship divided into 20 stations

# beam properties
beamSegments = 40

# section properties
verticalAreaMoments = np.ones([beamSegments]) * 0.59969e-05
horizontalAreaMoments = np.ones([beamSegments]) * 0.24325e-04
sectionalAreas = np.ones([beamSegments]) * 0.36945e-02
verticalShearAreaFractions = np.ones([beamSegments]) * 0.32148
horizontalShearAreaFractions = np.ones([beamSegments]) * 0.28385
zCentroidOverBottom = 0.45434e-01

# material properties
youngsModulus = 0.215e10
shearModulus = youngsModulus / (2 * (1 + 0.26))

# wave characteristics
waveHeight = 0.025

waterDepths = np.array([1.8])
omegas = np.array([4.50, 5.59, 5.81, 6.07, 6.37, 6.71, 7.12, 8.22])
waveDirections = np.array([0])

# mesh resolution
panelsLength = 100
panelsBreadth = 16
panelsDepth = 16



# CALCULATIONS

hullDraft = hullDisplacement / (waterDensity * hullLength * hullBreadth)
zNeutralAxis = -hullDraft + zCentroidOverBottom
centerOfMass = (0, 0, zNeutralAxis)

uniformlyDistributedMass = hullDisplacement - np.sum(pointMasses)
linearDensitiesBeam = np.ones([beamSegments]) * uniformlyDistributedMass / hullLength

beamDefinition = {}
beamDefinition['nodeXPositions'] = np.linspace(-hullLength/2, hullLength/2, beamSegments + 1)
beamDefinition['crossSectionAreas'] = sectionalAreas
beamDefinition['verticalAreaMoments'] = verticalAreaMoments
beamDefinition['horizontalAreaMoments'] = horizontalAreaMoments
beamDefinition['verticalTimoshenkoCoefs'] = verticalShearAreaFractions
beamDefinition['horizontalTimoshenkoCoefs'] = horizontalShearAreaFractions
beamDefinition['torsionConstants'] = np.ones([beamSegments])
beamDefinition['warpingConstants'] = np.ones([beamSegments])
beamDefinition['youngsModulus'] = youngsModulus
beamDefinition['shearModulus'] = shearModulus
beamDefinition['zNeutralAxis'] = zNeutralAxis
beamDefinition['zTwistCenter'] = zNeutralAxis
beamDefinition['linearDensities'] = linearDensitiesBeam

# beam is assumed to be parallel to the x axis and oriented towards its positive direction, i.e., the beam normal is [1,0,0]
beam = Beam(beamDefinition)

if not beamSegments % 40 == 0:
    sys.exit('Beam must be discretized in a number of segments that is a multiple of 40.')
segmentsPerHalfStation = beamSegments / 40
for i in range(pointMasses.size):
    vertex = int(pointMassStations[i] * 2 * segmentsPerHalfStation)

    beam.massMatrix[6 * vertex    , 6 * vertex    ] += pointMasses[i]
    beam.massMatrix[6 * vertex + 1, 6 * vertex + 1] += pointMasses[i]
    beam.massMatrix[6 * vertex + 2, 6 * vertex + 2] += pointMasses[i]


# dry natural frequencies
dryNaturalFrequenciesSquared, dryVibrationModes = eigh(beam.stiffnessMatrix, beam.massMatrix)

zeroFrequencyIndices = np.abs(dryNaturalFrequenciesSquared) < 0.01
dryNaturalFrequenciesSquared[zeroFrequencyIndices] = 0
dryNaturalFrequenciesHz = np.sqrt(dryNaturalFrequenciesSquared) / (2 * np.pi)

# mesh generation
hullSize = (hullLength, hullBreadth, hullDepth)
hullCenter = (0, 0, -hullDraft + hullDepth/2)
meshResolution = (panelsLength, panelsBreadth, panelsDepth)

hullMesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()

# creation of dofs from mesh and beam

dofs = beam.CalculateDOFs(hullMesh)

# definition of the body
hullBody = cpt.FloatingBody(mesh = hullMesh, dofs = dofs, center_of_mass = centerOfMass)

# solution of the array of problems
testMatrix = xr.Dataset(coords={
    'omega': omegas,
    'wave_direction': waveDirections,
    'radiating_dof': list(hullBody.dofs),
    'water_depth': waterDepths,
    'rho': waterDensity,
    'g' : gravity
})

# hydrostatic stiffness calculation
# hydrostaticStiffness = hullBody.compute_hydrostatic_stiffness(rho = waterDensity)
# hydrostaticStiffness.to_netcdf("data/hydrostatics.nc")
hydrostaticStiffness = xr.open_dataarray("data/hydrostatics.nc")

# hydrodynamic calculation: added mass, radiation, forcing
hydrodynamicResults = cpt.BEMSolver().fill_dataset(testMatrix, hullBody)
# hydrodynamicResults.to_netcdf("data/hydrodynamics.nc")
# hydrodynamicResults = xr.open_dataset("data/hydrodynamics.nc")

# coupling of hydrodynamic and structural results, springing results
springingResults = SpringingResults(beam.massMatrix, beam.stiffnessMatrix, hydrostaticStiffness, hydrodynamicResults)

# midships bending moments
midshipsBendingMoments = np.zeros([omegas.size], dtype = np.complex128)
midshipsBendingMomentAmplitudes = np.zeros([omegas.size])

for i in range(omegas.size):
    displacements = np.zeros([6*beam.numberNodes])
    displacements = waveHeight * springingResults.displacementAmplitudes.values[i, 0, :]
    midshipsBendingMoments[i] = beam.BendingMoment(hullLength/2, displacements)
    midshipsBendingMomentAmplitudes[i] = np.abs(midshipsBendingMoments[i])

bendingMomentCoefs = midshipsBendingMomentAmplitudes / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight)

# wavelengths
wavenumbers = omegas**2 / gravity
wavelengths = 2 * np.pi / wavenumbers



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


plt.figure()
plt.title('Midships bending moment coefficient for different waves')
plt.plot(hullLength/wavelengths, bendingMomentCoefs, 'ko')
plt.xlim([0,1.75])
plt.ylim([0, 0.03])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')

plt.show()


omegaIndex = 4
waveDirectionIndex = 0

motion = {}

dofIndex = 0

for dof in hullBody.dofs.keys():
    motion[dof] = waveHeight * springingResults.displacementAmplitudes.values[omegaIndex, waveDirectionIndex, dofIndex]
    dofIndex += 1

animation = hullBody.animate(motion = motion, loop_duration = 1)
animation.run()