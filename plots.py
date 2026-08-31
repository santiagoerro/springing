import numpy as np
import springing as spr
import os
import matplotlib.pyplot as plt



# INPUT

# water properties
waterDensity = 1000

# gravity
gravity = 9.81

# hull geometry
hullLength = 1.52
hullBreadth = 0.22
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
omegas = np.array([4.50, 5.59, 5.81, 6.07, 6.37, 6.71, 7.12, 8.22])


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

beam = spr.Beam(beamDefinition)


displacements20Modes = np.load('data/20modeDisplacements.npy')
displacements40Modes = np.load('data/40modeDisplacements.npy')
displacements6ModesProper = np.load('data/6modeProperDisplacements.npy')
displacementsNodal = np.load('data/nodalDisplacements.npy')

x = np.linspace(0, hullLength, 500)
bendingMomentDistribution20Modes = beam.InternalForce(x, displacements20Modes, 'mv')
shearForceDistribution20Modes = beam.InternalForce(x, displacements20Modes, 'sv')
bendingMomentDistribution40Modes = beam.InternalForce(x, displacements40Modes, 'mv')
shearForceDistribution40Modes = beam.InternalForce(x, displacements40Modes, 'sv')
bendingMomentDistribution6ModesProper = beam.InternalForce(x, displacements6ModesProper, 'mv')
shearForceDistribution6ModesProper = beam.InternalForce(x, displacements6ModesProper, 'sv')
bendingMomentDistributionNodal = beam.InternalForce(x, displacementsNodal, 'mv')
shearForceDistributionNodal = beam.InternalForce(x, displacementsNodal, 'sv')


allDisplacementsCoarseMesh = np.load('data/allNodalDisplacements.npy')
allDisplacementsFineMesh = np.load('data/allNodalDisplacementsFineMesh.npy')

# midships bending moments
midshipsBendingMomentCoefsCoarse = np.zeros([omegas.size])
midshipsBendingMomentCoefsFine = np.zeros([omegas.size])

for i in range(omegas.size):
    displacementsCoarse = waveHeight / 2 * allDisplacementsCoarseMesh[i, 0, :]
    displacementsFine = waveHeight / 2 * allDisplacementsFineMesh[i, 0, :]
    midshipsBendingMomentCoefsCoarse[i] = np.abs(beam.InternalForce(hullLength/2, displacementsCoarse, 'mv')) / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)
    midshipsBendingMomentCoefsFine[i] = np.abs(beam.InternalForce(hullLength/2, displacementsFine, 'mv')) / (waterDensity * gravity * hullLength**2 * hullBreadth * waveHeight/2)



# OUTPUT

if not os.path.exists('plots'):
    os.makedirs('plots')

fig, axs = plt.subplots(2, 2, figsize = (10, 8))
fig.suptitle('Vertical shear force distribution for omega = 6.07 rad/s.')
axs[0, 0].set_title('Standard method, 20 modes.')
axs[0, 0].plot(x, np.imag(shearForceDistribution20Modes), 'g', label = 'Imaginary part')
axs[0, 0].plot(x, np.real(shearForceDistribution20Modes), 'k', label = 'Real part')
axs[0, 0].set(ylabel = 'Vertical shear force [N]')
axs[0, 0].legend()
axs[0, 1].set_title('Standard method, 40 modes.')
axs[0, 1].plot(x, np.imag(shearForceDistribution40Modes), 'g', label = 'Imaginary part')
axs[0, 1].plot(x, np.real(shearForceDistribution40Modes), 'k', label = 'Real part')
axs[1, 0].set_title('Proposed alternate method, 6 modes.')
axs[1, 0].plot(x, np.imag(shearForceDistribution6ModesProper), 'g', label = 'Imaginary part')
axs[1, 0].plot(x, np.real(shearForceDistribution6ModesProper), 'k', label = 'Real part')
axs[1, 0].set(xlabel = 'x [m]', ylabel = 'Vertical shear force [N]')
axs[1, 1].set_title('Nodal solution.')
axs[1, 1].plot(x, np.imag(shearForceDistributionNodal), 'g', label = 'Imaginary part')
axs[1, 1].plot(x, np.real(shearForceDistributionNodal), 'k', label = 'Real part')
axs[1, 1].set(xlabel = 'x [m]')
fig.savefig('plots/shearForceModalComparison.png', dpi = 200, bbox_inches='tight')

fig, axs = plt.subplots(2, 2, figsize = (10, 8))
fig.suptitle('Vertical bending moment distribution for omega = 6.07 rad/s.')
axs[0, 0].set_title('Standard method, 20 modes.')
axs[0, 0].plot(x, np.imag(bendingMomentDistribution20Modes), 'g', label = 'Imaginary part')
axs[0, 0].plot(x, np.real(bendingMomentDistribution20Modes), 'k', label = 'Real part')
axs[0, 0].set(ylabel = 'Vertical bending moment [Nm]')
axs[0, 0].legend()
axs[0, 1].set_title('Standard method, 40 modes.')
axs[0, 1].plot(x, np.imag(bendingMomentDistribution40Modes), 'g', label = 'Imaginary part')
axs[0, 1].plot(x, np.real(bendingMomentDistribution40Modes), 'k', label = 'Real part')
axs[1, 0].set_title('Proposed alternate method, 6 modes.')
axs[1, 0].plot(x, np.imag(bendingMomentDistribution6ModesProper), 'g', label = 'Imaginary part')
axs[1, 0].plot(x, np.real(bendingMomentDistribution6ModesProper), 'k', label = 'Real part')
axs[1, 0].set(xlabel = 'x [m]', ylabel = 'Vertical bending moment [Nm]')
axs[1, 1].set_title('Nodal solution.')
axs[1, 1].plot(x, np.imag(bendingMomentDistributionNodal), 'g', label = 'Imaginary part')
axs[1, 1].plot(x, np.real(bendingMomentDistributionNodal), 'k', label = 'Real part')
axs[1, 1].set(xlabel = 'x [m]')
fig.savefig('plots/bendingMomentModalComparison.png', dpi = 200, bbox_inches = 'tight')


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

plt.figure()
plt.title('Midships bending moment coefficient for different waves')
plt.plot(Ll, midshipsBendingMomentCoefsCoarse, 'ko', label = 'Base mesh')
flag = True
for series in midshipsBendingMomentCoefsExperimental:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, midshipsBendingMomentCoefs2DNumerical, 'k--', label = '2D Hydroelascity')
plt.xlim([0,1.75])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('plots/baseMesh.png', dpi = 200)

plt.figure()
plt.title('Midships bending moment coefficient for different waves and two meshes')
plt.plot(Ll, midshipsBendingMomentCoefsCoarse, 'ko', label = 'Base mesh')
plt.plot(Ll, midshipsBendingMomentCoefsFine, 'rx', label = 'Fine mesh')
flag = True
for series in midshipsBendingMomentCoefsExperimental:
    if flag:
        flag = False
        plt.plot(Ll, series, 'bo', label = 'Experiment')
    else:
        plt.plot(Ll, series, 'bo')
plt.plot(Ll_full, midshipsBendingMomentCoefs2DNumerical, 'k--', label = '2D Hydroelascity')
plt.xlim([0,1.75])
plt.xlabel('Ship length / wavelength')
plt.ylabel('CM')
plt.legend()
plt.savefig('plots/convergence.png', dpi = 200)

plt.show()