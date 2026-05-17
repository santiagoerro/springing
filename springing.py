import sys
import numpy as np
import numpy.linalg as la
from scipy.linalg import eigh
import capytaine as cpt
import xarray as xr



class Beam:
    """
    Class used to define the beam that determines the deformations of the mesh and the structural behavior of the problem.
    """
    def __init__(self, beamDefinition: dict):
        """
        Instantiates a Beam variable according to the definition provided. The beam is assumed to have a straight neutral axis and
        center of twist axis, both parallel to the x-axis direction and and contained in the xz-plane. The beam's vertical position
        must be such that the z = 0 plane corresponds to the free surface of the water.

        :type beamDefinition: dict
        :param beamDefinition: Dictionary defining the beam properties, which must contain the values indexed by the following keys. Throughout, let `n` denote the number of FEM segments the beam is divided into.

            - `'nodeXPositions'`: `(n+1,)-numpy.ndarray` containing the x positions, in m, of the nodes that define the FEM mesh of
            the beam. It must contain at least two nodes, and all x values must be in strictly increasing order. The `i`-th segment
            is located between the `i` and `i+1` nodes.
            - `'crossSectionAreas'`: `(n,)-numpy.ndarray`, `i`-th component: structural cross sectional area of the `i`-th segment,
            in m^2.
            - `'verticalAreaMoments'`: `(n,)-numpy.ndarray`, `i`-th component: vertical second moment of area of the structural
            cross section of the `i`-th segment, in m^4.
            - `'horizontalAreaMoments'`: `(n,)-numpy.ndarray`, `i`-th component: horizontal second moment of area of the structural
            cross section of the `i`-th segment, in m^4.
            - `'verticalTimoshenkoCoefs'`: `(n,)-numpy.ndarray`, `i`-th component: vertical bending Timoshenko shear coefficient of
            the structural cross section of the `i`-th segment.
            - `'horizontalTimoshenkoCoefs'`: `(n,)-numpy.ndarray`, `i`-th component: horizontal bending Timoshenko shear coefficient
            of the structural cross section of the `i`-th segment.
            - `'torsionConstants'`: `(n,)-numpy.ndarray`, `i`-th component: torsion constant of the structural cross section of the
            `i`-th segment, in m^4.
            - `'warpingConstants'`: `(n,)-numpy.ndarray`, `i`-th component: warping constant of the structural cross section of the
            `i`-th segment, in m^6.
            - `'youngsModulus'`: `float`, the structural material's Young's modulus, in Pa.
            - `'shearModulus'`: `float`, the structural material's shear modulus, in Pa.
            - `'zNeutralAxis'`: `float`, the vertical position of the beam's neutral axis over the water's free surface, in m.
            - `'zTwistCenter'`: `float`, the vertical position of the beam's center of twist over the water's free surface, in m.
            - `'massMatrix'`: `(6*(n+1),6*(n+1))-numpy.ndarray` optional, FEM structural mass matrix of the beam, in SI units. To be provided
            if the user wishes to input it manually. Otherwise, `'linearDensities'`, `'zCentersOfMass'` and `'rollInertias'` must
            be provided.
            - `'linearDensities'`: `(n,)-numpy.ndarray` optional, `i`-th component: linear mass density of the `i`-th segment, in
            kg/m. Must be provided if the FEM structural mass matrix is not manually given by the user in `'massMatrix'`.
            - `'zCentersOfMass'`: `(n,)-numpy.ndarray` optional, `i`-th component: vertical position over the free surface of the
            center of mass of cross-sections on the `i`-th segment of the beam, in m. Must be provided if the FEM structural mass
            matrix is not manually given by the user in `'massMatrix'`.
            - `'rollInertias'`: `(n,)-numpy.ndarray` optional, `i`-th component: mass moment of inertia of the sections in the
            `i`-th segment around an axis parallel to the x-axis that passses through their center of mass, in kg m2 / m. Must be
            provided if the FEM structural mass matrix is not manually given by the user in `'massMatrix'`.
            - `'stiffnessMatrix'`: `(6*(n+1),6*(n+1))-numpy.ndarray` optional, FEM stiffness matrix of the beam, in SI units. To be provided
            if the user wishes to input it manually. If it is not provided, it will be calculated from the beam's structural properties.
        """

        self.nodeXPositions = beamDefinition['nodeXPositions']

        if not type(self.nodeXPositions) == np.ndarray:
            sys.exit('Invalid beam definition: nodeXPositions must be a numpy ndarray.')
        if not self.nodeXPositions.ndim == 1:
            sys.exit('Invalid beam definition: nodeXPositions must be a 1-dimensional array.')
        if self.nodeXPositions.size < 2:
            sys.exit('Invalid beam definition: the beam must consist of at least two nodes.')

        self.segmentLengths = np.diff(self.nodeXPositions)

        if np.any(self.segmentLengths <= 0):
            sys.exit('Invalid beam definition: node positions must be strictly increasing.')

        self.numberSegments = self.segmentLengths.size
        self.numberNodes = self.nodeXPositions.size


        def GetArrayCheck(dictionary: dict, key: str, shape: tuple) -> np.ndarray:
            array = dictionary[key]
            if not type(array) == np.ndarray:
                sys.exit('Invalid beam definition: %s must be a numpy ndarray.'%key)
            if not array.shape == shape:
                sys.exit('Invalid beam definition: %s must be a %s-array.'%(key, str(shape)))

            return array


        segmentsShape = self.segmentLengths.shape

        self.crossSectionAreas = GetArrayCheck(beamDefinition, 'crossSectionAreas', segmentsShape)
        self.verticalAreaMoments = GetArrayCheck(beamDefinition, 'verticalAreaMoments', segmentsShape)
        self.horizontalAreaMoments = GetArrayCheck(beamDefinition, 'horizontalAreaMoments', segmentsShape)
        self.verticalTimoshenkoCoefs = GetArrayCheck(beamDefinition, 'verticalTimoshenkoCoefs', segmentsShape)
        self.horizontalTimoshenkoCoefs = GetArrayCheck(beamDefinition, 'horizontalTimoshenkoCoefs', segmentsShape)
        self.torsionConstants = GetArrayCheck(beamDefinition, 'torsionConstants', segmentsShape)
        self.warpingConstants = GetArrayCheck(beamDefinition, 'warpingConstants', segmentsShape)

        self.youngsModulus = beamDefinition['youngsModulus']
        self.shearModulus = beamDefinition['shearModulus']

        self.zNeutralAxis = beamDefinition['zNeutralAxis']
        self.zTwistCenter = beamDefinition['zTwistCenter']

        self.verticalShearCorrections = 12 * self.youngsModulus * self.verticalAreaMoments / (self.verticalTimoshenkoCoefs * self.crossSectionAreas * self.shearModulus * self.segmentLengths**2)
        self.horizontalShearCorrections = 12 * self.youngsModulus * self.horizontalAreaMoments / (self.horizontalTimoshenkoCoefs * self.crossSectionAreas * self.shearModulus * self.segmentLengths**2)

        self.warpingWavenumbers = np.sqrt(self.shearModulus * self.torsionConstants / (self.youngsModulus * self.warpingConstants))
        self.warpingWavenumbersSegmentLengths = self.warpingWavenumbers * self.segmentLengths

        self.splineLimit = np.zeros([self.numberSegments], dtype = bool)
        self.boundedBasisCoefsForStableBasisW0 = np.zeros([self.numberSegments, 4])
        self.boundedBasisCoefsForStableBasisW1 = np.zeros([self.numberSegments, 4])

        for segment in range(0, self.numberSegments):
            a = self.warpingWavenumbersSegmentLengths[segment]

            if a < 0.003:
                self.splineLimit[segment] = True

            if a < 0.3:
                aSquare = a**2
                aCube = a**3
                aFourth = a**4
                aFifth = a**5
                aSixth = a**6
                aSeventh = a**7

                c11w0 = 4/aSquare + 2/15 - 11*aSquare/6300 + aFourth/27000 - 509*aSixth/582120000
                c12w0 = -6/aSquare - 1/10 + aSquare/1400 - aFourth/126000 + 37*aSixth/388080000
                c13w0 = 3/aCube + 1/aSquare + 1/(20*a) - 1/60 - a/2800 + 13*aSquare/25200 + aCube/252000 - 11*aFourth/756000 - 37*aFifth/776160000 + 907*aSixth/2328480000 + 59*aSeventh/100900800000
                c14w0 = -3/aCube - 2/aSquare - 11/(20*a) - 1/15 + a/2800 + 11*aSquare/12600 - aCube/252000 - aFourth/54000 + 37*aFifth/776160000 + 509*aSixth/1164240000 - 59*aSeventh/100900800000

                c11w1 = 2/aSquare - 1/30 + 13*aSquare/12600 - 11*aFourth/378000 + 907*aSixth/1164240000
                c13w1 = 3/aCube + 2/aSquare + 11/(20*a) + 1/15 - a/2800 - 11*aSquare/12600 + aCube/252000 + aFourth/54000 - 37*aFifth/776160000 - 509*aSixth/1164240000 + 59*aSeventh/100900800000
                c14w1 = -3/aCube - 1/aSquare - 1/(20*a) + 1/60 + a/2800 - 13*aSquare/25200 - aCube/252000 + 11*aFourth/756000 + 37*aFifth/776160000 - 907*aSixth/2328480000 - 59*aSeventh/100900800000
            elif a > 320:
                denominator = -a**2 + 2*a

                c11w0 = (-a + 1) / denominator
                c12w0 = -1 / (a - 2)
                c13w0 = -1 / denominator
                c14w0 = -c11w0

                c11w1 = -1 / denominator
                c13w1 = (-a + 1) / denominator
                c14w1 = -c11w1
            else:
                expa = np.exp(a)
                exp2a = np.exp(2*a)
                denominator = a**2 + 2*a - a**2*exp2a + 2*a*exp2a - 4*a*expa

                c11w0 = (-a - a*exp2a + exp2a - 1) / denominator
                c12w0 = (-expa + 1) / (a + a*expa - 2*expa + 2)
                c13w0 = (a*expa - exp2a + expa) / denominator
                c14w0 = (a*exp2a - exp2a + expa) / denominator

                c11w1 = (-exp2a + 2*a*expa + 1) / denominator
                c13w1 = (-a*exp2a + exp2a - expa) / denominator
                c14w1 = (exp2a - a*expa - expa) / denominator

            c12w1 = c12w0

            self.boundedBasisCoefsForStableBasisW0[segment, :] = np.array([c11w0, c12w0, c13w0, c14w0])
            self.boundedBasisCoefsForStableBasisW1[segment, :] = np.array([c11w1, c12w1, c13w1, c14w1])


        matrixShape = (6 * self.numberNodes, 6 * self.numberNodes)

        if 'massMatrix' in beamDefinition:
            self.massMatrix = GetArrayCheck(beamDefinition, 'massMatrix', matrixShape)
        else:
            if not 'linearDensities' in beamDefinition or not 'zCentersOfMass' in beamDefinition or not 'rollInertias' in beamDefinition:
                sys.exit('Invalid beam definition: if the mass matrix is not provided, linearDensities, zCentersOfMass and rollInertias must be specified for the mass matrix to be calculated.')
            linearDensities = GetArrayCheck(beamDefinition, 'linearDensities', segmentsShape)
            zCentersOfMass = GetArrayCheck(beamDefinition, 'zCentersOfMass', segmentsShape)
            rollInertias = GetArrayCheck(beamDefinition, 'rollInertias', segmentsShape)

            self.massMatrix = self.UniformlyDistributedMassMatrix(linearDensities, zCentersOfMass, rollInertias)

        if 'stiffnessMatrix' in beamDefinition:
            self.stiffnessMatrix = GetArrayCheck(beamDefinition, 'stiffnessMatrix', matrixShape)
        else:
            self.stiffnessMatrix = self.StiffnessMatrix()


    def UniformlyDistributedMassMatrix(self, linearDensities: np.ndarray, zCentersOfMass: np.ndarray, rollInertias: np.ndarray):
        """
        Creates a mass matrix for the Finite Elements Method, assuming linearly uniformly distributed
        mass within each segment. Throughout, `n` corresponds to the beam's number of nodes.

        :param linearDensities: Array. `i`-th component: linear mass density of the `i`-th segment of the beam, in kg/m.
        :type linearDensities: (n-1,)-numpy.ndarray

        :param zCentersOfMass: Array. `i`-th component: vertical position over the free surface of the center of mass of cross-sections on the `i`-th segment of the beam, in m.
        :type zCentersOfMass: (n-1,)-numpy.ndarray

        :param rollInertias: Array. `i`-th component: mass moment of inertia of the sections in the `i`-th segment around an axis parallel to the x-axis that passses through their center of mass, in kg m2 / m.
        :type rollInertias: (n-1,)-numpy.ndarray

        :returns: Mass matrix for the FEM analysis.
        :rtype: (7*n, 7*n)-numpy.ndarray
        """

        massMatrix = np.zeros([7 * self.numberNodes, 7 * self.numberNodes])

        x0 = 0
        y0 = 1
        z0 = 2
        r0 = 3
        w0 = 4
        tau0 = 5
        psi0 = 6
        x1 = 7
        y1 = 8
        z1 = 9
        r1 = 10
        w1 = 11
        tau1 = 12
        psi1 = 13

        assemblyBasisToStableBasisCoefsMatrix = np.eye(14)
        # in the assembly basis, the constrained warping roll 0 dof (which substitutes r0 in place) picks up +w0 and +w1
        assemblyBasisToStableBasisCoefsMatrix[w0, r0] = 1
        assemblyBasisToStableBasisCoefsMatrix[w1, r0] = 1
        # in the assembly basis, the constrained warping roll 1 dof (which substitutes r1 in place) picks up -w0 and -w1
        assemblyBasisToStableBasisCoefsMatrix[w0, r1] = -1
        assemblyBasisToStableBasisCoefsMatrix[w1, r1] = -1

        for i in range(self.numberSegments):
            segmentMassMatrixStableBasis = np.zeros([14, 14])

            segmentLength = self.segmentLengths[i]
            segmentMass = linearDensities[i] * segmentLength

            svi = self.verticalShearCorrections[i]
            svi2 = svi**2
            shi = self.horizontalShearCorrections[i]
            shi2 = shi**2

            # axial motion
            segmentMassMatrixStableBasis[x0,x0] = segmentMass / 3
            segmentMassMatrixStableBasis[x0,x1] = segmentMass / 6
            segmentMassMatrixStableBasis[x1,x0] = segmentMass / 6
            segmentMassMatrixStableBasis[x1,x1] = segmentMass / 3

            # vertical bending motion
            verticalFactor = segmentMass / (840 * (svi2 + 2 * svi + 1))

            segmentMassMatrixStableBasis[z0, z0] = verticalFactor * (280 * svi2 + 588 * svi + 312)
            segmentMassMatrixStableBasis[z0, z1] = verticalFactor * (140 * svi2 + 252 * svi + 108)
            segmentMassMatrixStableBasis[z1, z0] = verticalFactor * (140 * svi2 + 252 * svi + 108)
            segmentMassMatrixStableBasis[z1, z1] = verticalFactor * (280 * svi2 + 588 * svi + 312)

            segmentMassMatrixStableBasis[z0  , tau0] = verticalFactor * (-35 * svi2 - 77 * svi - 44) * segmentLength
            segmentMassMatrixStableBasis[tau0, z0  ] = verticalFactor * (-35 * svi2 - 77 * svi - 44) * segmentLength
            segmentMassMatrixStableBasis[z0  , tau1] = verticalFactor * ( 35 * svi2 + 63 * svi + 26) * segmentLength
            segmentMassMatrixStableBasis[tau1, z0  ] = verticalFactor * ( 35 * svi2 + 63 * svi + 26) * segmentLength
            segmentMassMatrixStableBasis[tau0, z1  ] = verticalFactor * (-35 * svi2 - 63 * svi - 26) * segmentLength
            segmentMassMatrixStableBasis[z1  , tau0] = verticalFactor * (-35 * svi2 - 63 * svi - 26) * segmentLength
            segmentMassMatrixStableBasis[z1  , tau1] = verticalFactor * ( 35 * svi2 + 77 * svi + 44) * segmentLength
            segmentMassMatrixStableBasis[tau1, z1  ] = verticalFactor * ( 35 * svi2 + 77 * svi + 44) * segmentLength

            segmentMassMatrixStableBasis[tau0, tau0] = verticalFactor * ( 7 * svi2 + 14 * svi + 8) * segmentLength**2
            segmentMassMatrixStableBasis[tau0, tau1] = verticalFactor * (-7 * svi2 - 14 * svi - 6) * segmentLength**2
            segmentMassMatrixStableBasis[tau1, tau0] = verticalFactor * (-7 * svi2 - 14 * svi - 6) * segmentLength**2
            segmentMassMatrixStableBasis[tau1, tau1] = verticalFactor * ( 7 * svi2 + 14 * svi + 8) * segmentLength**2

            # horizontal bending motion
            horizontalFactor = segmentMass / (840 * (shi2 + 2 * shi + 1))

            segmentMassMatrixStableBasis[y0, y0] = horizontalFactor * (280 * shi2 + 588 * shi + 312)
            segmentMassMatrixStableBasis[y0, y1] = horizontalFactor * (140 * shi2 + 252 * shi + 108)
            segmentMassMatrixStableBasis[y1, y0] = horizontalFactor * (140 * shi2 + 252 * shi + 108)
            segmentMassMatrixStableBasis[y1, y1] = horizontalFactor * (280 * shi2 + 588 * shi + 312)

            segmentMassMatrixStableBasis[y0  , psi0] = horizontalFactor * ( 35 * shi2 + 77 * shi + 44) * segmentLength
            segmentMassMatrixStableBasis[psi0, y0  ] = horizontalFactor * ( 35 * shi2 + 77 * shi + 44) * segmentLength
            segmentMassMatrixStableBasis[y0  , psi1] = horizontalFactor * (-35 * shi2 - 63 * shi - 26) * segmentLength
            segmentMassMatrixStableBasis[psi1, y0  ] = horizontalFactor * (-35 * shi2 - 63 * shi - 26) * segmentLength
            segmentMassMatrixStableBasis[psi0, y1  ] = horizontalFactor * ( 35 * shi2 + 63 * shi + 26) * segmentLength
            segmentMassMatrixStableBasis[y1  , psi0] = horizontalFactor * ( 35 * shi2 + 63 * shi + 26) * segmentLength
            segmentMassMatrixStableBasis[y1  , psi1] = horizontalFactor * (-35 * shi2 - 77 * shi - 44) * segmentLength
            segmentMassMatrixStableBasis[psi1, y1  ] = horizontalFactor * (-35 * shi2 - 77 * shi - 44) * segmentLength

            segmentMassMatrixStableBasis[psi0, psi0] = horizontalFactor * ( 7 * shi2 + 14 * shi + 8) * segmentLength**2
            segmentMassMatrixStableBasis[psi0, psi1] = horizontalFactor * (-7 * shi2 - 14 * shi - 6) * segmentLength**2
            segmentMassMatrixStableBasis[psi1, psi0] = horizontalFactor * (-7 * shi2 - 14 * shi - 6) * segmentLength**2
            segmentMassMatrixStableBasis[psi1, psi1] = horizontalFactor * ( 7 * shi2 + 14 * shi + 8) * segmentLength**2

            # torsional motion
            fullTwistInertiaL = (rollInertias[i] + linearDensities[i] * (zCentersOfMass[i] - self.zTwistCenter)**2) * segmentLength

            segmentMassMatrixStableBasis[r0, r0] = fullTwistInertiaL / 3
            segmentMassMatrixStableBasis[r0, r1] = fullTwistInertiaL / 6
            segmentMassMatrixStableBasis[r1, r0] = fullTwistInertiaL / 6
            segmentMassMatrixStableBasis[r1, r1] = fullTwistInertiaL / 3

            a = self.warpingWavenumbersSegmentLengths[i]

            if a <= 0:
                sys.exit('The product of the warping wavenumber and the segment length must not be zero on any segment.')
            elif a > 165:
                denominator = 6*a**3*(a**2 - 4 * a + 4)

                segmentMassMatrixStableBasis[w0, w0] = fullTwistInertiaL * (2*a**3 - 15*a**2 + 36*a - 18) / denominator
                segmentMassMatrixStableBasis[w0, w1] = fullTwistInertiaL * (-a**3 + 6*a**2 - 12*a + 18) / denominator

                segmentMassMatrixStableBasis[r0, w0] = fullTwistInertiaL * (2*a**2 - 9*a + 12)/(6*a**2*(a - 2))
                segmentMassMatrixStableBasis[r0, w1] = fullTwistInertiaL * (-a + 3)/(6*a*(a - 2))
            elif a < 0.68:
                segmentMassMatrixStableBasis[w0, w0] = fullTwistInertiaL * ( 1/105 - a**2/3150 + 149*a**4/14553000 - 361*a**6/1135134000 + 45691*a**8/4767562800000)
                segmentMassMatrixStableBasis[w0, w1] = fullTwistInertiaL * (-1/140 + a**2/3600 - 559*a**4/58212000 + 509*a**6/1651104000 - 13847*a**8/1466942400000)

                segmentMassMatrixStableBasis[r0, w0] = fullTwistInertiaL * ( 1/20 - 19*a**2/25200 + 13*a**4/756000 - 109*a**6/258720000 + 28703*a**8/2724321600000)
                segmentMassMatrixStableBasis[r0, w1] = fullTwistInertiaL * (-1/30 +    a**2/1575  -    a**4/63000  +  59*a**6/145530000 -  7043*a**8/681080400000)
            else:
                expa = np.exp(a)
                exp2a = np.exp(2 * a)
                exp3a = np.exp(3 * a)
                exp4a = np.exp(4 * a)
                denominator = (6*a**3*(a**2*exp4a - 2*a**2*exp2a + a**2 - 4*a*exp4a + 8*a*exp3a - 8*a*expa + 4*a + 4*exp4a - 16*exp3a + 24*exp2a - 16*expa + 4))

                segmentMassMatrixStableBasis[w0, w0] = fullTwistInertiaL * (2*a**3*exp4a + 4*a**3*exp3a + 24*a**3*exp2a + 4*a**3*expa + 2*a**3 - 15*a**2*exp4a - 24*a**2*exp3a + 24*a**2*expa + 15*a**2 + 36*a*exp4a - 36*a*exp3a - 36*a*expa + 36*a - 18*exp4a + 36*exp3a - 36*expa + 18) / denominator
                segmentMassMatrixStableBasis[w0, w1] = fullTwistInertiaL * (-a**3*exp4a - 14*a**3*exp3a - 6*a**3*exp2a - 14*a**3*expa - a**3 + 6*a**2*exp4a + 42*a**2*exp3a - 42*a**2*expa - 6*a**2 - 12*a*exp4a - 60*a*exp3a + 144*a*exp2a - 60*a*expa - 12*a + 18*exp4a - 36*exp3a + 36*expa - 18) / denominator

                segmentMassMatrixStableBasis[r0, w0] = fullTwistInertiaL * (2*a**2*exp2a + 2*a**2*expa + 2*a**2 - 9*a*exp2a + 9*a + 12*exp2a - 24*expa + 12)/(6*a**2*(a*exp2a - a - 2*exp2a + 4*expa - 2))
                segmentMassMatrixStableBasis[r0, w1] = fullTwistInertiaL * (-a*exp2a - 4*a*expa - a + 3*exp2a - 3)/(6*a*(a*exp2a - a - 2*exp2a + 4*expa - 2))

            segmentMassMatrixStableBasis[w1, w1] = segmentMassMatrixStableBasis[w0, w0]
            segmentMassMatrixStableBasis[w1, w0] = segmentMassMatrixStableBasis[w0, w1]

            segmentMassMatrixStableBasis[r1, w0] = -segmentMassMatrixStableBasis[r0, w1]
            segmentMassMatrixStableBasis[r1, w1] = -segmentMassMatrixStableBasis[r0, w0]

            segmentMassMatrixStableBasis[w0, r0] = segmentMassMatrixStableBasis[r0, w0]
            segmentMassMatrixStableBasis[w1, r0] = segmentMassMatrixStableBasis[r0, w1]
            segmentMassMatrixStableBasis[w0, r1] = segmentMassMatrixStableBasis[r1, w0]
            segmentMassMatrixStableBasis[w1, r1] = segmentMassMatrixStableBasis[r1, w1]
            
            segmentMassMatrixAssemblyBasis = assemblyBasisToStableBasisCoefsMatrix.transpose() @ segmentMassMatrixStableBasis @ assemblyBasisToStableBasisCoefsMatrix

            massMatrix[7 * i : 7 * (i + 2), 7 * i : 7 * (i + 2)] += segmentMassMatrixAssemblyBasis

        return massMatrix


    def StiffnessMatrix(self):
        """
        Calculates the beam's Finite Elements Method stiffness matrix according to Timoshenko bending and Vlasov thin-walled torsion beam theory. `n` corresponds to the beam's number of nodes.

        :returns: Beam's FEM stiffness matrix.
        :rtype: (7*n, 7*n)-numpy.ndarray
        """

        stiffnessMatrix = np.zeros([7 * self.numberNodes, 7 * self.numberNodes])

        x0 = 0
        y0 = 1
        z0 = 2
        r0 = 3
        w0 = 4
        tau0 = 5
        psi0 = 6
        x1 = 7
        y1 = 8
        z1 = 9
        r1 = 10
        w1 = 11
        tau1 = 12
        psi1 = 13

        assemblyBasisToStableBasisCoefsMatrix = np.eye(14)
        # in the assembly basis, the constrained warping roll 0 dof (which substitutes r0 in place) picks up +w0 and +w1
        assemblyBasisToStableBasisCoefsMatrix[w0, r0] = 1
        assemblyBasisToStableBasisCoefsMatrix[w1, r0] = 1
        # in the assembly basis, the constrained warping roll 1 dof (which substitutes r1 in place) picks up -w0 and -w1
        assemblyBasisToStableBasisCoefsMatrix[w0, r1] = -1
        assemblyBasisToStableBasisCoefsMatrix[w1, r1] = -1

        for i in range(self.numberSegments):
            segmentStiffnessMatrixStableBasis = np.zeros([14, 14])

            segmentLength = self.segmentLengths[i]
            shearCorrectionVertical = self.verticalShearCorrections[i]
            shearCorrectionHorizontal = self.horizontalShearCorrections[i]

            # axial stiffness
            EAOverL = self.youngsModulus * self.crossSectionAreas[i] / segmentLength

            segmentStiffnessMatrixStableBasis[x0, x0] = EAOverL
            segmentStiffnessMatrixStableBasis[x0, x1] = -EAOverL
            segmentStiffnessMatrixStableBasis[x1, x0] = -EAOverL
            segmentStiffnessMatrixStableBasis[x1, x1] = EAOverL

            # vertical bending
            EIOverCorrection = self.youngsModulus * self.verticalAreaMoments[i] / (1 + shearCorrectionVertical)

            segmentStiffnessMatrixStableBasis[z0, z0] = 12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[z0, z1] = -12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[z1, z0] = -12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[z1, z1] = 12 * EIOverCorrection / segmentLength**3

            segmentStiffnessMatrixStableBasis[z0  , tau0] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[tau0, z0  ] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[z0  , tau1] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[tau1, z0  ] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[tau0, z1  ] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[z1  , tau0] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[z1  , tau1] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[tau1, z1  ] = 6 * EIOverCorrection / segmentLength**2

            segmentStiffnessMatrixStableBasis[tau0, tau0] = (4 + shearCorrectionVertical) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[tau0, tau1] = (2 - shearCorrectionVertical) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[tau1, tau0] = (2 - shearCorrectionVertical) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[tau1, tau1] = (4 + shearCorrectionVertical) * EIOverCorrection / segmentLength

            # horizontal bending
            EIOverCorrection = self.youngsModulus * self.horizontalAreaMoments[i] / (1 + shearCorrectionHorizontal)

            segmentStiffnessMatrixStableBasis[y0, y0] = 12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[y0, y1] = -12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[y1, y0] = -12 * EIOverCorrection / segmentLength**3
            segmentStiffnessMatrixStableBasis[y1, y1] = 12 * EIOverCorrection / segmentLength**3

            segmentStiffnessMatrixStableBasis[y0  , psi0] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[psi0, y0  ] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[y0  , psi1] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[psi1, y0  ] = 6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[psi0, y1  ] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[y1  , psi0] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[y1  , psi1] = -6 * EIOverCorrection / segmentLength**2
            segmentStiffnessMatrixStableBasis[psi1, y1  ] = -6 * EIOverCorrection / segmentLength**2

            segmentStiffnessMatrixStableBasis[psi0, psi0] = (4 + shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[psi0, psi1] = (2 - shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[psi1, psi0] = (2 - shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            segmentStiffnessMatrixStableBasis[psi1, psi1] = (4 + shearCorrectionHorizontal) * EIOverCorrection / segmentLength

            # torsional stiffness
            GTorsionOverL = self.shearModulus * self.torsionConstants[i] / segmentLength

            segmentStiffnessMatrixStableBasis[r0, r0] = GTorsionOverL
            segmentStiffnessMatrixStableBasis[r0, r1] = -GTorsionOverL
            segmentStiffnessMatrixStableBasis[r1, r0] = -GTorsionOverL
            segmentStiffnessMatrixStableBasis[r1, r1] = GTorsionOverL

            EWarpingOverL3 = self.youngsModulus * self.warpingConstants[i] / segmentLength**3
            a = self.warpingWavenumbersSegmentLengths[i]

            if a <= 0:
                sys.exit('The product of the warping number and the segment length must not be zero on any segment.')
            elif a > 320:
                segmentStiffnessMatrixStableBasis[w0, w0] = EWarpingOverL3 * (-a**2 + a) / (-a + 2)
                segmentStiffnessMatrixStableBasis[w0, w1] = EWarpingOverL3 * (-a) / (-a + 2)
            elif a < 0.08:
                segmentStiffnessMatrixStableBasis[w0, w0] = EWarpingOverL3 * (4 + 2/15 * a**2 - 11/6300  * a**4 + 1/27000   * a**6)
                segmentStiffnessMatrixStableBasis[w0, w1] = EWarpingOverL3 * (2 - 1/30 * a**2 + 13/12600 * a**4 - 11/378000 * a**6)
            else:
                expa = np.exp(a)
                exp2a = np.exp(2 * a)
                denominator = a - a * exp2a + 2 * exp2a - 4 * expa + 2
                segmentStiffnessMatrixStableBasis[w0, w0] = EWarpingOverL3 * (-a**2 - a - a**2 * exp2a + a * exp2a) / denominator
                segmentStiffnessMatrixStableBasis[w0, w1] = EWarpingOverL3 * (a - a * exp2a + 2 * a**2 * expa) / denominator

            segmentStiffnessMatrixStableBasis[w1, w0] = segmentStiffnessMatrixStableBasis[w0, w1]
            segmentStiffnessMatrixStableBasis[w1, w1] = segmentStiffnessMatrixStableBasis[w0, w0]

            segmentStiffnessMatrixAssemblyBasis = assemblyBasisToStableBasisCoefsMatrix.transpose() @ segmentStiffnessMatrixStableBasis @ assemblyBasisToStableBasisCoefsMatrix

            stiffnessMatrix[7 * i : 7 * (i + 2), 7 * i : 7 * (i + 2)] += segmentStiffnessMatrixAssemblyBasis

        return stiffnessMatrix


    def SegmentInternalForce(self, xSegment: float | np.ndarray, segmentDisplacements: np.ndarray, segmentIndex: int, force: str):
        """
        TODO: Update.
        Calculates the vertical or horizontal bending moments at a one or more points within a beam segment
        given their x coordinates within the segment and the displacements of the nodes around it.

        :param xSegment: Value or array of values of the x coordinate within the beam segment, in meters, of
            the point or points at which the bending moment is to be calculated.
        :type xSegment: float or (m,)-numpy.ndarray

        :param segmentDisplacements: Array of displacements, in meters and radians, of the beam nodes before
            and after the beam segment at which the bending moment is to be calculated.
        :type segmentDisplacements: (12,)-numpy.ndarray

        :param areaMoment: Vertical or horizontal geometric area moment of the beam section of the segment of interest,
            in m^4.
        :type areaMoment: float

        :param shearCorrection: Vertical or horizontal shear correction of the segment of interest.
        :type shearCorrection: float

        :param plane: String identifying if the vertical, `'v'`, or horizontal, `'h'`, bending moment is to
            be calculated.
        :type plane: str

        :returns: Value or array of values of the vertical or horizontal bending moment at the point or points
            indicated by `xSegment`, in Nm.
        :rtype: float or (m,)-numpy.ndarray
        """
        segmentLength = self.segmentLengths[segmentIndex]

        if type(xSegment) == np.ndarray:
            if np.min(xSegment) < 0 or np.max(xSegment) > segmentLength:
                sys.exit('All values for the x coordinate within the beam segment must be between 0 and the segment length.')
        else:
            if xSegment < 0 or xSegment > segmentLength:
                sys.exit('The value for the x coordinate within the beam segment must be between 0 and the segment length.')

        if force == 'a':
            x0 = segmentDisplacements[0]
            x1 = segmentDisplacements[7]

            if not type(xSegment) == np.ndarray:
                return self.youngsModulus * self.crossSectionAreas[segmentIndex] / segmentLength * (x1 - x0)
            else:
                return self.youngsModulus * self.crossSectionAreas[segmentIndex] / segmentLength * (x1 - x0) * np.ones_like(xSegment)

        elif force == 'mv':
            z0 = segmentDisplacements[2]
            tau0 = segmentDisplacements[5]
            z1 = segmentDisplacements[9]
            tau1 = segmentDisplacements[12]

            svi = self.verticalShearCorrections[segmentIndex]

            polynomialZ0 = -12 * xSegment/segmentLength + 6
            polynomialTau0 = 6 * xSegment               - (4 + svi) * segmentLength
            polynomialZ1 =  12 * xSegment/segmentLength - 6
            polynomialTau1 = 6 * xSegment               + (-2 + svi) * segmentLength

            prefactor = self.youngsModulus * self.verticalAreaMoments[segmentIndex] / ((1 + svi) * segmentLength**2)

            return prefactor * (polynomialZ0 * z0 + polynomialTau0 * tau0 + polynomialZ1 * z1 + polynomialTau1 * tau1)

        elif force == 'sv':
            z0 = segmentDisplacements[2]
            tau0 = segmentDisplacements[5]
            z1 = segmentDisplacements[9]
            tau1 = segmentDisplacements[12]

            svi = self.verticalShearCorrections[segmentIndex]

            prefactor = self.youngsModulus * self.verticalAreaMoments[segmentIndex] / ((1 + svi) * segmentLength**3)

            shearForceValue = prefactor * (-12 * z0 + 6*segmentLength * tau0 + 12 * z1 + 6*segmentLength * tau1)

            if not type(xSegment) == np.ndarray:
                return shearForceValue
            else:
                return shearForceValue * np.ones_like(xSegment)

        elif force == 'mh':
            y0 = segmentDisplacements[1]
            psi0 = segmentDisplacements[6]
            y1 = segmentDisplacements[8]
            psi1 = segmentDisplacements[13]

            shi = self.horizontalShearCorrections[segmentIndex]

            polynomialY0 =  12 * xSegment/segmentLength - 6
            polynomialPsi0 = 6 * xSegment               - (4 + shi) * segmentLength
            polynomialY1 = -12 * xSegment/segmentLength + 6
            polynomialPsi1 = 6 * xSegment               + (-2 + shi) * segmentLength

            prefactor = self.youngsModulus * self.horizontalAreaMoments[segmentIndex] / ((1 + shi) * segmentLength**2)

            return prefactor * (polynomialY0 * y0 + polynomialPsi0 * psi0 + polynomialY1 * y1 + polynomialPsi1 * psi1)

        elif force == 'sh':
            y0 = segmentDisplacements[1]
            psi0 = segmentDisplacements[6]
            y1 = segmentDisplacements[8]
            psi1 = segmentDisplacements[13]

            shi = self.horizontalShearCorrections[segmentIndex]

            prefactor = self.youngsModulus * self.horizontalAreaMoments[segmentIndex] / ((1 + shi) * segmentLength**3)

            shearForceValue = prefactor * (-12 * z0 - 6*segmentLength * tau0 + 12 * z1 - 6*segmentLength * tau1)

            if not type(xSegment) == np.ndarray:
                return shearForceValue
            else:
                return shearForceValue * np.ones_like(xSegment)

        elif force == 't':
            phi0 = segmentDisplacements[3]
            w0 = segmentDisplacements[4]
            phi1 = segmentDisplacements[10]
            w1 = segmentDisplacements[11]

            prefactor = self.shearModulus * self.torsionConstants[segmentIndex] / segmentLength

            r0TorsionMoment = -prefactor
            r1TorsionMoment =  prefactor
            w0TorsionMoment = prefactor * self.boundedBasisCoefsForStableBasisW0[segmentIndex, 1]
            w1TorsionMoment = w0TorsionMoment

            phi0TorsionMoment = r0TorsionMoment + w0TorsionMoment + w1TorsionMoment
            phi1TorsionMoment = r1TorsionMoment - w0TorsionMoment - w1TorsionMoment

            torsionMomentValue = phi0 * phi0TorsionMoment + w0 * w0TorsionMoment + phi1 * phi1TorsionMoment + w1 * w1TorsionMoment

            if not type(xSegment) == np.ndarray:
                return torsionMomentValue
            else:
                return torsionMomentValue * np.ones_like(xSegment)

        elif force == 'tf' or force == 'tw':
            phi0 = segmentDisplacements[3]
            w0 = segmentDisplacements[4]
            phi1 = segmentDisplacements[10]
            w1 = segmentDisplacements[11]

            k = self.warpingWavenumbers[segmentIndex]
            C12w0 = self.boundedBasisCoefsForStableBasisW0[segmentIndex, 1]
            C13w0 = self.boundedBasisCoefsForStableBasisW0[segmentIndex, 2]
            C14w0 = self.boundedBasisCoefsForStableBasisW0[segmentIndex, 3]
            C12w1 = self.boundedBasisCoefsForStableBasisW1[segmentIndex, 1]
            C13w1 = self.boundedBasisCoefsForStableBasisW1[segmentIndex, 2]
            C14w1 = self.boundedBasisCoefsForStableBasisW1[segmentIndex, 3]

            prefactor = self.shearModulus * self.torsionConstants[segmentIndex]
            positiveExponentialBoundedBasis = np.exp( k * (xSegment - segmentLength))
            negativeExponentialBoundedBasis = np.exp(-k * xSegment)

            if force == 'tf':
                r0TorsionMoment = -prefactor / segmentLength
                r1TorsionMoment =  prefactor / segmentLength
                w0TorsionMoment = prefactor * (C12w0 / segmentLength + C13w0 * k * positiveExponentialBoundedBasis - C14w0 * k * negativeExponentialBoundedBasis)
                w1TorsionMoment = prefactor * (C12w1 / segmentLength + C13w1 * k * positiveExponentialBoundedBasis - C14w1 * k * negativeExponentialBoundedBasis)
            else:
                r0TorsionMoment = 0
                r1TorsionMoment = 0
                w0TorsionMoment = -prefactor * k * (C13w0 * positiveExponentialBoundedBasis - C14w0* negativeExponentialBoundedBasis)
                w1TorsionMoment = -prefactor * k * (C13w1 * positiveExponentialBoundedBasis - C14w1* negativeExponentialBoundedBasis)

            phi0TorsionMoment = r0TorsionMoment + w0TorsionMoment + w1TorsionMoment
            phi1TorsionMoment = r1TorsionMoment - w0TorsionMoment - w1TorsionMoment

            torsionMomentValue = phi0 * phi0TorsionMoment + w0 * w0TorsionMoment + phi1 * phi1TorsionMoment + w1 * w1TorsionMoment

            return torsionMomentValue

        else:
            sys.exit('TODO: force must be one of [etc].')


    def InternalForce(self, x: float | np.ndarray, displacements: np.ndarray, force: str):
        """
        TODO: Update.
        Calculates the vertical or horizontal bending moments at a one or more points within the beam
        given their x coordinates within the beam and the full vector of vertex displacements. Throughout,
        n corresponds to the beam's number of nodes.

        :param x: Value or array of values of the x coordinate within the beam, in meters, of the point
            or points at which the bending moment is to be calculated.
        :type x: float or (m,)-numpy.ndarray

        :param displacements: Array of displacements, in meters and radians, of all of the beam nodes.
        :type displacements: (6*n,)-numpy.ndarray

        :param plane: String identifying if the vertical, `'v'`, or horizontal, `'h'`, bending moment is to
            be calculated.
        :type plane: str, optional. Default: `'v'`.

        :returns: Value or array of values of the vertical or horizontal bending moment at the point or points
            indicated by `x`, in Nm.
        :rtype: float or (m,)-numpy.ndarray
        """
        if not force in ['a', 'mv', 'sv', 'mh', 'sh', 't', 'tf', 'tw']:
            sys.exit('TODO: force must be one of [etc].')

        if not self.numberNodes * 7 == displacements.size:
            sys.exit('Wrong size of displacements vector')

        nodeXPositionsStartingAtZero = self.nodeXPositions - self.nodeXPositions[0]
        beamLength = nodeXPositionsStartingAtZero[-1]

        if not type(x) == np.ndarray:
            if x < 0 or x > beamLength:
                sys.exit('The position x at which the bending moment will be calculated must be between 0 and the beam length.')

            aftVertexIndex = np.searchsorted(nodeXPositionsStartingAtZero, x, side = 'right')-1
            coordinateSegment = x - nodeXPositionsStartingAtZero[aftVertexIndex]

            segmentDisplacements = displacements[7*aftVertexIndex : 7*(aftVertexIndex + 2)]

            return self.SegmentInternalForce(coordinateSegment, segmentDisplacements, aftVertexIndex, force)


        if not np.all(np.sort(x) == x):
            sys.exit('The array of x positions at which the bending moment will be calculated must be sorted.')

        if x[0] < 0 or x[-1] > beamLength:
            sys.exit('All positions x at which the bending moment will be calculated must be between 0 and the beam length.')

        internalForceDistribution = np.zeros([x.size], dtype = displacements.dtype)

        for i in range(self.numberSegments):
            if i == self.numberSegments - 1:
                mask = (x >= nodeXPositionsStartingAtZero[i]) & (x <= beamLength)
            else:
                mask = (x >= nodeXPositionsStartingAtZero[i]) & (x < nodeXPositionsStartingAtZero[i+1])

            coordinatesSegment = x[mask] - nodeXPositionsStartingAtZero[i]

            if not coordinatesSegment.size == 0:
                segmentDisplacements = displacements[7*i : 7*(i + 2)]

                internalForceDistribution[mask] = self.SegmentInternalForce(coordinatesSegment, segmentDisplacements, i, force)

        return internalForceDistribution


    def CalculateNodalDOFs(self, hullMesh : cpt.Mesh):
        """
        Creates the degrees of freedom corresponding to the object's mesh and beam.

        This function takes the object's mesh and beam and creates Capytaine degrees of freedom.
        To each beam vertex, six degrees of freedom are associated: three linear and three angular,
        according to the right-hand rule sign convention. The degrees of freedom corresponding to
        the `i`-th vertex are labeled by the following strings.

        Surge `'x%d'%i`
        
        Sway `'y%d'%i`
        
        Heave `'z%d'%i`
        
        Roll `'roll%d'%i`
        
        Pitch `'pitch%d'%i`
        
        Yaw `'yaw%d'%i`

        The beam's neutral axis is deformed between nodes according to the FEM interpolation
        polynomials for beam theory. That is, cubic for bending, linear for axial and torsional.
        This extends to a deformation of the whole mesh under the assumption that sections remain
        perpendicular to the neutral axis. The field of displacements is linearized, as required by
        hydrodynamic panel code formulations. The mesh regions before the first and after the last
        beam nodes are deformed as rigid bodies fixed to the endpoint vertex.

        :type hullMesh: capytaine.Mesh
        :param hullMesh: Mesh of the ship's hull, positioned in the same coordinate system as the beam. In particular, the hull must be located in such a way that hull girder's neutral axis and center of
        twist are parallel to the x-axis and contained in the xz-plane. The z = 0 plane must coincide with the water's free surface.

        :rtype: dict
        :returns: A dictionary containing the degrees of freedom, ready for input to Capytaine.
        """

        self.segmentOfFace = np.searchsorted(self.nodeXPositions, hullMesh.faces_centers[:, 0])-1

        self.facesOnSegment = np.ndarray([self.numberSegments + 2], dtype=object)

        for i in range(-1, self.numberSegments + 1):
            self.facesOnSegment[i] = np.where(self.segmentOfFace == i)[0]

        self.yFaces = np.zeros([hullMesh.nb_faces])
        self.zFaces = np.zeros([hullMesh.nb_faces])
        self.yFaces = hullMesh.faces_centers[:, 1]
        self.zFaces = hullMesh.faces_centers[:, 2]

        self.xSegmentFaces = np.zeros([hullMesh.nb_faces])
        self.chiSegmentFaces = np.zeros([hullMesh.nb_faces])

        # Segment -1: before the start of the beam
        xInitialSegment = self.nodeXPositions[0]

        for face in self.facesOnSegment[-1]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = -1

        # Segments 0 to self.numberSegments - 1: within the beam
        for segment in range(0, self.numberSegments):
            xInitialSegment = self.nodeXPositions[segment]

            for face in self.facesOnSegment[segment]:
                self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
                self.chiSegmentFaces[face] = self.xSegmentFaces[face] / self.segmentLengths[segment]

        # Segment self.numberSegments: after the end of the beam
        xInitialSegment = self.nodeXPositions[self.numberSegments]

        for face in self.facesOnSegment[self.numberSegments]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = 2

        dofs = {}

        for vertex in range(self.numberNodes):
            surgeDofName = 'x%d'%vertex # axial
            swayDofName  = 'y%d'%vertex # bending horizontal
            heaveDofName = 'z%d'%vertex # bending vertical

            rollDofName  = 'roll%d'%vertex      # torsion
            warpingDofName = 'warping%d'%vertex # torsion
            pitchDofName = 'pitch%d'%vertex     # bending horizontal
            yawDofName   = 'yaw%d'%vertex       # bending vertical

            surgeDofDisplacements = np.zeros([hullMesh.nb_faces, 3])
            swayDofDisplacements  = np.zeros([hullMesh.nb_faces, 3])
            heaveDofDisplacements = np.zeros([hullMesh.nb_faces, 3])

            rollDofDisplacements    = np.zeros([hullMesh.nb_faces, 3])
            warpingDofDisplacements = np.zeros([hullMesh.nb_faces, 3])
            pitchDofDisplacements   = np.zeros([hullMesh.nb_faces, 3])
            yawDofDisplacements     = np.zeros([hullMesh.nb_faces, 3])


            # segment before the vertex
            segment = vertex - 1

            if segment == -1:
                for face in self.facesOnSegment[segment]:
                    surgeDofDisplacements[face,:] = np.array([1, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([0, 1, 0])
                    heaveDofDisplacements[face,:] = np.array([0, 0, 1])

                    rollDofDisplacements[face,:]  = np.array([0,                                     -(self.zFaces[face] - self.zTwistCenter), self.yFaces[face]        ])
                    pitchDofDisplacements[face,:] = np.array([self.zFaces[face] - self.zNeutralAxis, 0,                                        -self.xSegmentFaces[face]])
                    yawDofDisplacements[face,:]   = np.array([-self.yFaces[face],                    self.xSegmentFaces[face],                 0                        ])
            else:
                segmentLength = self.segmentLengths[segment]

                for face in self.facesOnSegment[segment]:
                    chiSegment = self.chiSegmentFaces[face]
                    yFace = self.yFaces[face]
                    zFaceFromNeutralAxis = self.zFaces[face] - self.zNeutralAxis
                    zFaceFromTwistCenter = self.zFaces[face] - self.zTwistCenter

                    sv = self.verticalShearCorrections[segment]
                    sh = self.horizontalShearCorrections[segment]
                    a = self.warpingWavenumbersSegmentLengths[segment]

                    verticalBendingDeflectionHeave = (sv * chiSegment + 3 * chiSegment**2 - 2 * chiSegment**3) / (1 + sv)
                    verticalBendingRotationHeave = (-6 * chiSegment + 6 * chiSegment**2) / segmentLength / (1 + sv)
                    verticalBendingDeflectionPitch = (sv/2 * chiSegment + (2 - sv)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + sv)
                    verticalBendingRotationPitch = ((-2 + sv) * chiSegment + 3 * chiSegment**2) / (1 + sv)

                    horizontalBendingDeflectionSway = (sh * chiSegment + 3 * chiSegment**2 - 2 * chiSegment**3) / (1 + sh)
                    horizontalBendingRotationSway = -(-6 * chiSegment + 6 * chiSegment**2) / (1 + sh) / segmentLength
                    horizontalBendingDeflectionYaw = -(sh/2 * chiSegment + (2 - sh)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + sh)
                    horizontalBendingRotationYaw = ((-2 + sh) * chiSegment + 3 * chiSegment**2) / (1 + sh)

                    if self.splineLimit[segment]:
                        twistAngleWarpingAft = chiSegment * (chiSegment - 1)**2
                        twistAngleWarping = chiSegment**2 * (chiSegment - 1)
                    else:
                        boundedBasis = np.array([1, chiSegment, np.exp(a * (chiSegment - 1)), np.exp(-a * chiSegment)])

                        twistAngleWarpingAft = np.dot(boundedBasis, self.boundedBasisCoefsForStableBasisW0[segment, :])
                        twistAngleWarping = np.dot(boundedBasis, self.boundedBasisCoefsForStableBasisW1[segment, :])

                    twistAngleRoll = chiSegment - twistAngleWarpingAft - twistAngleWarping

                    surgeDofDisplacements[face,:] = np.array([chiSegment, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([-yFace * horizontalBendingRotationSway, horizontalBendingDeflectionSway, 0])
                    heaveDofDisplacements[face,:] = np.array([ zFaceFromNeutralAxis * verticalBendingRotationHeave, 0, verticalBendingDeflectionHeave])

                    rollDofDisplacements[face,:]    = np.array([0, -zFaceFromTwistCenter * twistAngleRoll, yFace * twistAngleRoll])
                    warpingDofDisplacements[face,:] = np.array([0, -zFaceFromTwistCenter * twistAngleWarping, yFace * twistAngleWarping])
                    pitchDofDisplacements[face,:]   = np.array([ zFaceFromNeutralAxis * verticalBendingRotationPitch, 0, verticalBendingDeflectionPitch])
                    yawDofDisplacements[face,:]     = np.array([-yFace * horizontalBendingRotationYaw, horizontalBendingDeflectionYaw, 0])

            # segment after the vertex
            segment = vertex

            if segment == self.numberSegments:
                for face in self.facesOnSegment[segment]:
                    surgeDofDisplacements[face,:] = np.array([1, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([0, 1, 0])
                    heaveDofDisplacements[face,:] = np.array([0, 0, 1])

                    rollDofDisplacements[face,:]  = np.array([0,                                     -(self.zFaces[face] - self.zTwistCenter), self.yFaces[face]        ])
                    pitchDofDisplacements[face,:] = np.array([self.zFaces[face] - self.zNeutralAxis, 0,                                        -self.xSegmentFaces[face]])
                    yawDofDisplacements[face,:]   = np.array([-self.yFaces[face],                    self.xSegmentFaces[face],                 0                        ])
            else:
                segmentLength = self.segmentLengths[segment]

                for face in self.facesOnSegment[segment]:
                    chiSegment = self.chiSegmentFaces[face]
                    yFace = self.yFaces[face]
                    zFaceFromNeutralAxis = self.zFaces[face] - self.zNeutralAxis
                    zFaceFromTwistCenter = self.zFaces[face] - self.zTwistCenter

                    sv = self.verticalShearCorrections[segment]
                    sh = self.horizontalShearCorrections[segment]
                    a = self.warpingWavenumbersSegmentLengths[segment]

                    verticalBendingDeflectionHeave = 1 + (-sv * chiSegment - 3 * chiSegment**2 + 2 * chiSegment**3) / (1 + sv)
                    verticalBendingRotationHeave = (6 * chiSegment - 6 * chiSegment**2) / segmentLength / (1 + sv)
                    verticalBendingDeflectionPitch = ((-2 - sv)/2 * chiSegment + (4 + sv)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + sv)
                    verticalBendingRotationPitch = 1 + ((-4 - sv) * chiSegment + 3 * chiSegment**2) / (1 + sv)

                    horizontalBendingDeflectionSway = 1 + (-sh * chiSegment - 3 * chiSegment**2 + 2 * chiSegment**3) / (1 + sh)
                    horizontalBendingRotationSway = -(6 * chiSegment - 6 * chiSegment**2) / segmentLength / (1 + sh)
                    horizontalBendingDeflectionYaw = -((-2 - sh)/2 * chiSegment + (4 + sh)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + sh)
                    horizontalBendingRotationYaw = 1 + ((-4 - sv) * chiSegment + 3 * chiSegment**2) / (1 + sv)

                    if self.splineLimit[segment]:
                        twistAngleWarping = chiSegment * (chiSegment - 1)**2
                        twistAngleWarpingFore = chiSegment**2 * (chiSegment - 1)
                    else:
                        boundedBasis = np.array([1, chiSegment, np.exp(a * (chiSegment - 1)), np.exp(-a * chiSegment)])

                        twistAngleWarping = np.dot(boundedBasis, self.boundedBasisCoefsForStableBasisW0[segment, :])
                        twistAngleWarpingFore = np.dot(boundedBasis, self.boundedBasisCoefsForStableBasisW1[segment, :])

                    twistAngleRoll = 1 - chiSegment + twistAngleWarping + twistAngleWarpingFore

                    surgeDofDisplacements[face,:] = np.array([1 - chiSegment, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([-yFace * horizontalBendingRotationSway, horizontalBendingDeflectionSway, 0])
                    heaveDofDisplacements[face,:] = np.array([ zFaceFromNeutralAxis * verticalBendingRotationHeave, 0, verticalBendingDeflectionHeave])

                    rollDofDisplacements[face,:]    = np.array([0, -zFaceFromTwistCenter * twistAngleRoll, yFace * twistAngleRoll])
                    warpingDofDisplacements[face,:] = np.array([0, -zFaceFromTwistCenter * twistAngleWarping, yFace * twistAngleWarping])
                    pitchDofDisplacements[face,:]   = np.array([ zFaceFromNeutralAxis * verticalBendingRotationPitch, 0, verticalBendingDeflectionPitch])
                    yawDofDisplacements[face,:]     = np.array([-yFace * horizontalBendingRotationYaw, horizontalBendingDeflectionYaw, 0])

            dofs[surgeDofName] = surgeDofDisplacements
            dofs[swayDofName]  = swayDofDisplacements
            dofs[heaveDofName] = heaveDofDisplacements

            dofs[rollDofName]    = rollDofDisplacements
            dofs[warpingDofName] = warpingDofDisplacements
            dofs[pitchDofName]   = pitchDofDisplacements
            dofs[yawDofName]     = yawDofDisplacements

        return dofs


    def CalculateModalDOFs(self, hullMesh: cpt.Mesh, numberModes: int, rigidBodyModesFrequencySquaredTolerance = 1e-3):
        if numberModes < 6:
            sys.exit('numberModes must be greater than or equal to 6: at least the rigid body modes must be considered.')

        allDryNaturalFrequenciesSquared, allDryVibrationModesNormalized = eigh(self.stiffnessMatrix, self.massMatrix)

        sortingIndices = np.argsort(np.abs(allDryNaturalFrequenciesSquared))

        if not np.abs(allDryNaturalFrequenciesSquared[sortingIndices[5]]) < np.abs(allDryNaturalFrequenciesSquared[sortingIndices[6]]) * rigidBodyModesFrequencySquaredTolerance:
            sys.exit('Rigid body mode frequencies are not sufficiently smaller than flexible mode frequencies, according to the tolerance provided.')

        allDryNaturalFrequenciesSquared[sortingIndices[0:6]] = np.zeros([6])

        initialIndex = np.min(sortingIndices[0:6])
        if not initialIndex == 0:
            print('Warning: Imaginary dry natural frequencies were encountered. There could be inconsistencies in the definition of the mass or stiffness matrices.')

        numberNodalDOFs = self.stiffnessMatrix.shape[0]

        if initialIndex + numberModes > numberNodalDOFs:
            sys.exit('The requested number of modes is greater than the number of available real frequency modes.')

        requestedDryNaturalFrequenciesSquared = np.zeros([numberModes])
        requestedDryVibrationModesNormalized = np.zeros([numberNodalDOFs, numberModes])

        requestedDryNaturalFrequenciesSquared = allDryNaturalFrequenciesSquared[initialIndex : initialIndex + numberModes]
        requestedDryVibrationModesNormalized = allDryVibrationModesNormalized[:, initialIndex : initialIndex + numberModes]

        nodalDofs = self.CalculateNodalDOFs(hullMesh)

        modalDofs = {}

        for i in range(numberModes):
            modeDisplacements = np.zeros([hullMesh.nb_faces, 3])
            nodalDofIndex = 0

            for nodalDof in nodalDofs.keys():
                modeDisplacements = modeDisplacements + nodalDofs[nodalDof] * requestedDryVibrationModesNormalized[nodalDofIndex, i]
                nodalDofIndex = nodalDofIndex + 1
            
            modeName = 'mode%d'%i
            modalDofs[modeName] = modeDisplacements
        
        return requestedDryNaturalFrequenciesSquared, requestedDryVibrationModesNormalized, modalDofs



class NodalSpringingResults:
    """
    Class used to compute and store the results of the ship's springing analysis.
    """
    def __init__(self, massMatrix: np.ndarray, stiffnessMatrix: np.ndarray, hydrostaticStiffness: xr.DataArray, hydrodynamicResults: xr.Dataset):
        """
        Instantiates a SpringingResults variable, calculating and storing the results of the springing
        analysis defined by the parameters passed to it. Throughout, n corresponds to the beam's number
        of nodes.

        :param massMatrix: Structural mass matrix of the ship, as a Finite Elements Method mass
            matrix for the hull girder beam.
        :type massMatrix: (6*n,6*n)-numpy.ndarray

        :param stiffnessMatrix: Structural stiffness matrix of the ship, as a Finite Elements
            Method stiffness matrix for the hull girder beam.
        :type stiffnessMatrix: (6*n,6*n)-numpy.ndarray

        :param hydrostaticStiffness: Capytaine hydrostatic stiffness results for the FloatingBody
            defined by the ship, as returned by the `compute_hydrostatic_stiffness` method of
            the `capytaine.FloatingBody` class. The FloatingBody must include the degrees of
            freedom given by the beam vertex motions, as calculated by the `CreateDOFs` method
            of the `MeshBeamProperties` class.
        :type hydrostaticStiffness: xarray.DataArray

        :param hydrodynamicResults: Dataset of Capytaine linear potential flow results for the
            FloatingBody defined by the ship on a test matrix with different wave frequencies,
            directions and water depths, as returned by the `fill_dataset` method of the
            `capytaine.BEMSolver` class. The FloatingBody must include the degrees of freedom
            given by the beam vertex motions, as calculated by the `MeshBeamProperties.CreateDOFs`
            method.
        :type hydrodynamicResults: xarray.Dataset

        :returns: Class object containing the following attributes.

            * displacementAmplitudes (xarray.DataArray): Array of results for the complex amplitudes
                of the springing motions of the beam nodes divided by wave height, in m/m and
                rad/m. The motions are sinusoidal, with the real part of the amplitude being the
                displacement at t = 0 and the imaginary part being the displacement, with
                opposite sign, after one fourth of the period. The array's dimensions are labeled
                `'omega'`, `'wave_direction'`, `'water_depth'` and `'dof'`. The length and
                coordinates associated to each of the first three dimensions match those of the
                `added_mass` and `radiation_damping` attributes of `hydrodynamicResults`, and
                are determined by the test matrix passed to the Capytaine BEM Solver when calculating
                these results. The `'dof'` dimension has a length of 6n and indexes the degree of
                freedom each amplitude corresponds to.
            * massMatrix (xarray.DataArray): The provided `massMatrix` as an `xarray.DataArray`,
                with dimensions labeled `'influenced_dof'` and `'radiating_dof'`.
            * stiffnessMatrix (xarray.DataArray): The provided `stiffnessMatrix` as an `xarray.DataArray`,
                with dimensions labeled `'influenced_dof'` and `'radiating_dof'`.
        :rtype: SpringingResults
        """
        self.massMatrix = xr.DataArray(massMatrix, dims = ['influenced_dof', 'radiating_dof'])
        self.stiffnessMatrix = xr.DataArray(stiffnessMatrix, dims = ['influenced_dof', 'radiating_dof'])

        self.forcesFromAmplitudesMatrices: xr.DataArray = - (hydrodynamicResults.added_mass + self.massMatrix) * hydrodynamicResults.omega**2 + complex(0,1) * hydrodynamicResults.omega * hydrodynamicResults.radiation_damping + (self.stiffnessMatrix + hydrostaticStiffness)

        self.amplitudesFromForcesMatrices = xr.DataArray(la.inv(self.forcesFromAmplitudesMatrices), dims = ['omega', 'radiating_dof', 'influenced_dof'])

        self.displacementAmplitudes: xr.DataArray = xr.dot(hydrodynamicResults.excitation_force, self.amplitudesFromForcesMatrices, dims = ['influenced_dof'])
        self.displacementAmplitudes = self.displacementAmplitudes.rename({'radiating_dof': 'dof'})

        self.hydrostaticStiffness = hydrostaticStiffness
        self.hydrodynamicResults = hydrodynamicResults



class ModalSpringingResults:
    """
    Class used to compute and store the results of the ship's modal springing analysis.
    """
    def __init__(self, dryNaturalFrequenciesSquared: np.ndarray, modalHydrostaticStiffness: xr.DataArray, modalHydrodynamicResults: xr.Dataset):
        """
        Instantiates a SpringingResults variable, calculating and storing the results of the springing
        analysis defined by the parameters passed to it. Throughout, n corresponds to the beam's number
        of nodes.

        :param massMatrix: Structural mass matrix of the ship, as a Finite Elements Method mass
            matrix for the hull girder beam.
        :type massMatrix: (6*n,6*n)-numpy.ndarray

        :param stiffnessMatrix: Structural stiffness matrix of the ship, as a Finite Elements
            Method stiffness matrix for the hull girder beam.
        :type stiffnessMatrix: (6*n,6*n)-numpy.ndarray

        :param hydrostaticStiffness: Capytaine hydrostatic stiffness results for the FloatingBody
            defined by the ship, as returned by the `compute_hydrostatic_stiffness` method of
            the `capytaine.FloatingBody` class. The FloatingBody must include the degrees of
            freedom given by the beam vertex motions, as calculated by the `CreateDOFs` method
            of the `MeshBeamProperties` class.
        :type hydrostaticStiffness: xarray.DataArray

        :param hydrodynamicResults: Dataset of Capytaine linear potential flow results for the
            FloatingBody defined by the ship on a test matrix with different wave frequencies,
            directions and water depths, as returned by the `fill_dataset` method of the
            `capytaine.BEMSolver` class. The FloatingBody must include the degrees of freedom
            given by the beam vertex motions, as calculated by the `MeshBeamProperties.CreateDOFs`
            method.
        :type hydrodynamicResults: xarray.Dataset

        :returns: Class object containing the following attributes.

            * displacementAmplitudes (xarray.DataArray): Array of results for the complex amplitudes
                of the springing motions of the beam nodes divided by wave height, in m/m and
                rad/m. The motions are sinusoidal, with the real part of the amplitude being the
                displacement at t = 0 and the imaginary part being the displacement, with
                opposite sign, after one fourth of the period. The array's dimensions are labeled
                `'omega'`, `'wave_direction'`, `'water_depth'` and `'dof'`. The length and
                coordinates associated to each of the first three dimensions match those of the
                `added_mass` and `radiation_damping` attributes of `hydrodynamicResults`, and
                are determined by the test matrix passed to the Capytaine BEM Solver when calculating
                these results. The `'dof'` dimension has a length of 6n and indexes the degree of
                freedom each amplitude corresponds to.
            * massMatrix (xarray.DataArray): The provided `massMatrix` as an `xarray.DataArray`,
                with dimensions labeled `'influenced_dof'` and `'radiating_dof'`.
            * stiffnessMatrix (xarray.DataArray): The provided `stiffnessMatrix` as an `xarray.DataArray`,
                with dimensions labeled `'influenced_dof'` and `'radiating_dof'`.
        :rtype: SpringingResults
        """
        massMatrix = np.eye(dryNaturalFrequenciesSquared.size)
        self.massMatrix = xr.DataArray(massMatrix, dims = ['influenced_dof', 'radiating_dof'])
        stiffnessMatrix = np.diag(dryNaturalFrequenciesSquared)
        self.stiffnessMatrix = xr.DataArray(stiffnessMatrix, dims = ['influenced_dof', 'radiating_dof'])

        self.modalForcesFromAmplitudesMatrices: xr.DataArray = - (modalHydrodynamicResults.added_mass + self.massMatrix) * modalHydrodynamicResults.omega**2 + complex(0,1) * modalHydrodynamicResults.omega * modalHydrodynamicResults.radiation_damping + (self.stiffnessMatrix + modalHydrostaticStiffness)

        self.modalAmplitudesFromForcesMatrices = xr.DataArray(la.inv(self.modalForcesFromAmplitudesMatrices), dims = ['omega', 'radiating_dof', 'influenced_dof'])

        self.modalAmplitudes: xr.DataArray = xr.dot(modalHydrodynamicResults.excitation_force, self.modalAmplitudesFromForcesMatrices, dims = ['influenced_dof'])
        self.modalAmplitudes = self.modalAmplitudes.rename({'radiating_dof': 'dof'})

        self.hydrostaticStiffness = modalHydrostaticStiffness
        self.hydrodynamicResults = modalHydrodynamicResults