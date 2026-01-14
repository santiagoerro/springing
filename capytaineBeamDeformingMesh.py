import sys
import numpy as np
import numpy.linalg as la
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
            if the user wishes to input it manually.
            - `'linearDensities'`: `(n,)-numpy.ndarray` optional, `i`-th component: linear mass density of the `i`-th segment, in
            kg/m. Must be provided if the FEM structural mass matrix is not manually given by the user in `'massMatrix'`. The FEM
            mass matrix is then calculated from these linear mass densities.
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


        matrixShape = (6 * self.numberNodes, 6 * self.numberNodes)

        if 'massMatrix' in beamDefinition:
            self.massMatrix = GetArrayCheck(beamDefinition, 'massMatrix', matrixShape)
        else:
            if not 'linearDensities' in beamDefinition:
                sys.exit('Invalid beam definition: if the mass matrix is not provided, linearDensities must be specified for the mass matrix to be calculated.')
            linearDensities = GetArrayCheck(beamDefinition, 'linearDensities', segmentsShape)

            self.massMatrix = self.UniformlyDistributedMassMatrix(linearDensities)

        if 'stiffnessMatrix' in beamDefinition:
            self.stiffnessMatrix = GetArrayCheck(beamDefinition, 'stiffnessMatrix', matrixShape)
        else:
            self.stiffnessMatrix = self.StiffnessMatrix()


    def UniformlyDistributedMassMatrix(self, linearDensities :np.ndarray):
        """
        Creates a mass matrix for the Finite Elements Method, assuming linearly uniformly distributed
        mass within each segment. Throughout, `n` corresponds to the beam's number of nodes.

        :param linearDensities: Array. `i`-th component: linear mass density of the `i`-th segment of the beam, in kg/m.
        :type linearDensities: (n-1,)-numpy.ndarray

        :returns: Mass matrix for the FEM analysis.
        :rtype: (6*n, 6*n)-numpy.ndarray
        """
        if linearDensities.size != self.numberSegments:
            sys.exit('Invalid linear densities definition: number of linear densities provided does not match number of beam segments.')

        massMatrix = np.zeros([self.numberNodes * 6, self.numberNodes * 6])

        for i in range(self.numberSegments):
            segmentLength = self.segmentLengths[i]
            segmentMassOver420 = linearDensities[i] * segmentLength / 420

            x1 = 6*i
            y1 = 6*i + 1
            z1 = 6*i + 2
            phi1 = 6*i + 3
            tau1 = 6*i + 4
            psi1 = 6*i + 5
            x2 = 6*i + 6
            y2 = 6*i + 7
            z2 = 6*i + 8
            phi2 = 6*i + 9
            tau2 = 6*i + 10
            psi2 = 6*i + 11

            # axial motion
            massMatrix[x1,x1] += 140 * segmentMassOver420
            massMatrix[x1,x2] += 70 * segmentMassOver420
            massMatrix[x2,x1] += 70 * segmentMassOver420
            massMatrix[x2,x2] += 140 * segmentMassOver420

            # vertical bending motion
            massMatrix[z1, z1] += 156 * segmentMassOver420
            massMatrix[z1, z2] += 54 * segmentMassOver420
            massMatrix[z2, z1] += 54 * segmentMassOver420
            massMatrix[z2, z2] += 156 * segmentMassOver420

            massMatrix[z1  , tau1] += -22 * segmentLength * segmentMassOver420
            massMatrix[tau1, z1  ] += -22 * segmentLength * segmentMassOver420
            massMatrix[z1  , tau2] += 13 * segmentLength * segmentMassOver420
            massMatrix[tau2, z1  ] += 13 * segmentLength * segmentMassOver420
            massMatrix[tau1, z2  ] += -13 * segmentLength * segmentMassOver420
            massMatrix[z2  , tau1] += -13 * segmentLength * segmentMassOver420
            massMatrix[z2  , tau2] += 22 * segmentLength * segmentMassOver420
            massMatrix[tau2, z2  ] += 22 * segmentLength * segmentMassOver420

            massMatrix[tau1, tau1] += 4 * segmentLength**2 * segmentMassOver420
            massMatrix[tau1, tau2] += -3 * segmentLength**2 * segmentMassOver420
            massMatrix[tau2, tau1] += -3 * segmentLength**2 * segmentMassOver420
            massMatrix[tau2, tau2] += 4 * segmentLength**2 * segmentMassOver420

            # horizontal bending motion
            massMatrix[y1, y1] += 156 * segmentMassOver420
            massMatrix[y1, y2] += 54 * segmentMassOver420
            massMatrix[y2, y1] += 54 * segmentMassOver420
            massMatrix[y2, y2] += 156 * segmentMassOver420

            massMatrix[y1  , psi1] += 22 * segmentLength * segmentMassOver420
            massMatrix[psi1, y1  ] += 22 * segmentLength * segmentMassOver420
            massMatrix[y1  , psi2] += -13 * segmentLength * segmentMassOver420
            massMatrix[psi2, y1  ] += -13 * segmentLength * segmentMassOver420
            massMatrix[psi1, y2  ] += 13 * segmentLength * segmentMassOver420
            massMatrix[y2  , psi1] += 13 * segmentLength * segmentMassOver420
            massMatrix[y2  , psi2] += -22 * segmentLength * segmentMassOver420
            massMatrix[psi2, y2  ] += -22 * segmentLength * segmentMassOver420

            massMatrix[psi1, psi1] += 4 * segmentLength**2 * segmentMassOver420
            massMatrix[psi1, psi2] += -3 * segmentLength**2 * segmentMassOver420
            massMatrix[psi2, psi1] += -3 * segmentLength**2 * segmentMassOver420
            massMatrix[psi2, psi2] += 4 * segmentLength**2 * segmentMassOver420

            # torsional motion: TODO
            massMatrix[phi1, phi1] = 2 * segmentMassOver420
            massMatrix[phi1, phi2] = 1 * segmentMassOver420
            massMatrix[phi2, phi1] = 1 * segmentMassOver420
            massMatrix[phi2, phi2] = 2 * segmentMassOver420
        
        return massMatrix


    def StiffnessMatrix(self):
        """
        Calculates the beam's Finite Elements Method stiffness matrix according to Timoshenko beam theory. `n` corresponds to the beam's number of nodes.

        :returns: Beam's FEM stiffness matrix.
        :rtype: (6*n, 6*n)-numpy.ndarray
        """

        stiffnessMatrix = np.zeros([self.numberNodes * 6, self.numberNodes * 6])

        for i in range(self.numberSegments):
            segmentLength = self.segmentLengths[i]
            shearCorrectionVertical = self.verticalShearCorrections[i]
            shearCorrectionHorizontal = self.horizontalShearCorrections[i]

            x1 = 6*i
            y1 = 6*i + 1
            z1 = 6*i + 2
            phi1 = 6*i + 3
            tau1 = 6*i + 4
            psi1 = 6*i + 5
            x2 = 6*i + 6
            y2 = 6*i + 7
            z2 = 6*i + 8
            phi2 = 6*i + 9
            tau2 = 6*i + 10
            psi2 = 6*i + 11

            # axial stiffness
            EAOverL = self.youngsModulus * self.crossSectionAreas[i] / segmentLength

            stiffnessMatrix[x1, x1] += EAOverL
            stiffnessMatrix[x1, x2] += -EAOverL
            stiffnessMatrix[x2, x1] += -EAOverL
            stiffnessMatrix[x2, x2] += EAOverL

            # vertical bending
            EIOverCorrection = self.youngsModulus * self.verticalAreaMoments[i] / (1 + shearCorrectionVertical)

            stiffnessMatrix[z1, z1] += 12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[z1, z2] += -12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[z2, z1] += -12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[z2, z2] += 12 * EIOverCorrection / segmentLength**3

            stiffnessMatrix[z1  , tau1] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[tau1, z1  ] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[z1  , tau2] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[tau2, z1  ] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[tau1, z2  ] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[z2  , tau1] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[z2  , tau2] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[tau2, z2  ] += 6 * EIOverCorrection / segmentLength**2

            stiffnessMatrix[tau1, tau1] += (4 + shearCorrectionVertical) * EIOverCorrection / segmentLength
            stiffnessMatrix[tau1, tau2] += (2 - shearCorrectionVertical) * EIOverCorrection / segmentLength
            stiffnessMatrix[tau2, tau1] += (2 - shearCorrectionVertical) * EIOverCorrection / segmentLength
            stiffnessMatrix[tau2, tau2] += (4 + shearCorrectionVertical) * EIOverCorrection / segmentLength

            # horizontal bending
            EIOverCorrection = self.youngsModulus * self.horizontalAreaMoments[i] / (1 + shearCorrectionHorizontal)

            stiffnessMatrix[y1, y1] += 12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[y1, y2] += -12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[y2, y1] += -12 * EIOverCorrection / segmentLength**3
            stiffnessMatrix[y2, y2] += 12 * EIOverCorrection / segmentLength**3

            stiffnessMatrix[y1  , psi1] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[psi1, y1  ] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[y1  , psi2] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[psi2, y1  ] += 6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[psi1, y2  ] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[y2  , psi1] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[y2  , psi2] += -6 * EIOverCorrection / segmentLength**2
            stiffnessMatrix[psi2, y2  ] += -6 * EIOverCorrection / segmentLength**2

            stiffnessMatrix[psi1, psi1] += (4 + shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            stiffnessMatrix[psi1, psi2] += (2 - shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            stiffnessMatrix[psi2, psi1] += (2 - shearCorrectionHorizontal) * EIOverCorrection / segmentLength
            stiffnessMatrix[psi2, psi2] += (4 + shearCorrectionHorizontal) * EIOverCorrection / segmentLength

            stiffnessMatrix[phi1, phi1] = EAOverL
            stiffnessMatrix[phi1, phi2] = -EAOverL
            stiffnessMatrix[phi2, phi1] = -EAOverL
            stiffnessMatrix[phi2, phi2] = EAOverL

        return stiffnessMatrix


    def SegmentBendingMoment(self, xSegment, segmentLength, segmentDisplacements, areaMoment, shearCorrection, plane):
        """
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
        if type(xSegment) == np.ndarray:
            if np.min(xSegment) < 0 or np.max(xSegment) > segmentLength:
                sys.exit('All values for the x coordinate within the beam segment must be between 0 and the segment length.')
        else:
            if xSegment < 0 or xSegment > segmentLength:
                sys.exit('The value for the x coordinate within the beam segment must be between 0 and the segment length.')

        if plane == 'v':
            linear1 = segmentDisplacements[2]
            angular1 = -segmentDisplacements[4]
            linear2 = segmentDisplacements[8]
            angular2 = -segmentDisplacements[10]
            #TODO: these minus signs beak the code because the DOFs and the stiffness and mass matrices follow different sign criteria. They have to be recalculated... Be smart about it. Relate one to the other to make it easy. But check symbolically.
        elif plane == 'h':
            linear1 = segmentDisplacements[1]
            angular1 = segmentDisplacements[5]
            linear2 = segmentDisplacements[7]
            angular2 = segmentDisplacements[11]
        else:
            sys.exit('Bending moment must be vertical (v) or horizontal (h).')

        L = segmentLength

        polynomialLinear1 =  (12*xSegment/L**3 -   6/L**2                ) / (1 + shearCorrection)
        polynomialAngular1 =  (6*xSegment/L**2 -  (4 + shearCorrection)/L) / (1 + shearCorrection)
        polynomialLinear2 = (-12*xSegment/L**3 +   6/L**2                ) / (1 + shearCorrection)
        polynomialAngular2 =  (6*xSegment/L**2 + (-2 + shearCorrection)/L) / (1 + shearCorrection)

        bendingMoment = self.youngsModulus * areaMoment * (polynomialLinear1 * linear1 + polynomialAngular1 * angular1 + polynomialLinear2  * linear2 + polynomialAngular2 * angular2)

        return bendingMoment


    def BendingMoment(self, x, displacements :np.ndarray, plane = 'v'):
        """
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
        if not (plane == 'v' or plane == 'h'):
            sys.exit('Bending moment must be vertical (v) or horizontal (h).')

        if not self.numberNodes*6 == displacements.size:
            sys.exit('Wrong size of displacements vector')

        nodeXPositionsStartingAtZero = self.nodeXPositions - self.nodeXPositions[0]
        beamLength = nodeXPositionsStartingAtZero[-1]

        if type(x) == np.ndarray:
            if not np.sort(x) == x:
                sys.exit('The array of x positions at which the bending moment will be calculated must be sorted.')

            if x[0] < 0 or x[-1] > beamLength:
                sys.exit('All positions x at which the bending moment will be calculated must be between 0 and the beam length.')
        else:
            if x < 0 or x > beamLength:
                sys.exit('The position x at which the bending moment will be calculated must be between 0 and the beam length.')

            leftVertexIndex = np.searchsorted(nodeXPositionsStartingAtZero, x, side = 'right')-1
            segmentLength = self.segmentLengths[leftVertexIndex]
            coordinateSegment = x - nodeXPositionsStartingAtZero[leftVertexIndex]

            segmentDisplacements = displacements[6*leftVertexIndex : 6*(leftVertexIndex + 2)]

            if plane == 'v':
                verticalAreaMoment = self.verticalAreaMoments[leftVertexIndex]
                verticalShearCorrection = self.verticalShearCorrections[leftVertexIndex]
                return self.SegmentBendingMoment(coordinateSegment, segmentLength, segmentDisplacements, verticalAreaMoment, verticalShearCorrection, 'v')
            else:
                horizontalAreaMoment = self.horizontalAreaMoments[leftVertexIndex]
                horizontalShearCorrection = self.horizontalShearCorrections[leftVertexIndex]
                return self.SegmentBendingMoment(coordinateSegment, segmentLength, segmentDisplacements, horizontalAreaMoment, horizontalShearCorrection, 'h')

        bendingMomentDistribution = np.zeros([x.size], dtype = np.complex128)

        for i in range(self.numberSegments):
            if i == self.numberSegments - 1:
                mask = (x >= nodeXPositionsStartingAtZero[i]) & (x <= beamLength)
            else:
                mask = (x >= nodeXPositionsStartingAtZero[i]) & (x < nodeXPositionsStartingAtZero[i+1])

            coordinatesSegment = x[mask] - nodeXPositionsStartingAtZero[i]

            if not coordinatesSegment.size == 0:
                segmentDisplacements = displacements[6*i : 6*(i + 2)]
                segmentLength = self.segmentLengths[i]

                if plane == 'v':
                    verticalAreaMoment = self.verticalAreaMoments[i]
                    verticalShearCorrection = self.verticalShearCorrections[i]
                    bendingMomentDistribution[mask] = self.SegmentBendingMoment(coordinatesSegment, segmentLength, segmentDisplacements, verticalAreaMoment, verticalShearCorrection, 'v')
                else:
                    horizontalAreaMoment = self.horizontalAreaMoments[i]
                    horizontalShearCorrection = self.horizontalShearCorrections[i]
                    bendingMomentDistribution[mask] = self.SegmentBendingMoment(coordinatesSegment, segmentLength, segmentDisplacements, horizontalAreaMoment, horizontalShearCorrection, 'h')

        return bendingMomentDistribution
    

    def CalculateDOFs(self, hullMesh : cpt.Mesh):
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


        self.xSegmentFaces = np.zeros([hullMesh.nb_faces])
        self.chiSegmentFaces = np.zeros([hullMesh.nb_faces])
        self.yOffsetFaces = np.zeros([hullMesh.nb_faces])
        self.zOffsetFaces = np.zeros([hullMesh.nb_faces])

        # Segment -1: before the start of the beam
        xInitialSegment = self.nodeXPositions[0]

        for face in self.facesOnSegment[-1]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = -1

            self.yOffsetFaces[face] = hullMesh.faces_centers[face,1]
            self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - self.zNeutralAxis

        # Segments 0 to self.numberSegments - 1: within the beam
        for segment in range(0, self.numberSegments):
            xInitialSegment = self.nodeXPositions[segment]

            for face in self.facesOnSegment[segment]:
                self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
                self.chiSegmentFaces[face] = self.xSegmentFaces[face] / self.segmentLengths[segment]

                self.yOffsetFaces[face] = hullMesh.faces_centers[face,1]
                self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - self.zNeutralAxis

        # Segment self.numberSegments: after the end of the beam
        xInitialSegment = self.nodeXPositions[self.numberSegments]

        for face in self.facesOnSegment[self.numberSegments]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = 2

            self.yOffsetFaces[face] = hullMesh.faces_centers[face,1]
            self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - self.zNeutralAxis

        dofs = {}

        for vertex in range(self.numberNodes):
            surgeDofName = 'x%d'%vertex # axial
            swayDofName  = 'y%d'%vertex # bending horizontal
            heaveDofName = 'z%d'%vertex # bending vertical

            rollDofName  = 'roll%d'%vertex  # torsion
            pitchDofName = 'pitch%d'%vertex # bending horizontal
            yawDofName   = 'yaw%d'%vertex   # bending vertical

            surgeDofDisplacements = np.zeros([hullMesh.nb_faces, 3])
            swayDofDisplacements  = np.zeros([hullMesh.nb_faces, 3])
            heaveDofDisplacements = np.zeros([hullMesh.nb_faces, 3])

            rollDofDisplacements  = np.zeros([hullMesh.nb_faces, 3])
            pitchDofDisplacements = np.zeros([hullMesh.nb_faces, 3])
            yawDofDisplacements   = np.zeros([hullMesh.nb_faces, 3])


            # segment before the vertex
            segment = vertex - 1

            if segment == -1:
                for face in self.facesOnSegment[segment]:
                    surgeDofDisplacements[face,:] = np.array([1, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([0, 1, 0])
                    heaveDofDisplacements[face,:] = np.array([0, 0, 1])

                    rollDofDisplacements[face,:]  = np.array([0,                        -self.zOffsetFaces[face], self.yOffsetFaces[face]  ])
                    pitchDofDisplacements[face,:] = np.array([self.zOffsetFaces[face],  0,                        -self.xSegmentFaces[face]])
                    yawDofDisplacements[face,:]   = np.array([-self.yOffsetFaces[face], self.xSegmentFaces[face], 0                        ])
            else:
                segmentLength = self.segmentLengths[segment]

                for face in self.facesOnSegment[segment]:
                    chiSegment = self.chiSegmentFaces[face]
                    yOffset = self.yOffsetFaces[face]
                    zOffset = self.zOffsetFaces[face]

                    deltaV = self.verticalShearCorrections[segment]
                    deltaH = self.horizontalShearCorrections[segment]

                    verticalBendingDeflectionHeave = (deltaV * chiSegment + 3 * chiSegment**2 - 2 * chiSegment**3) / (1 + deltaV)
                    verticalBendingRotationHeave = (-6 * chiSegment + 6 * chiSegment**2) / segmentLength / (1 + deltaV)
                    verticalBendingDeflectionPitch = (deltaV/2 * chiSegment + (2 - deltaV)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + deltaV)
                    verticalBendingRotationPitch = ((-2 + deltaV) * chiSegment + 3 * chiSegment**2) / (1 + deltaV)

                    horizontalBendingDeflectionSway = (deltaH * chiSegment + 3 * chiSegment**2 - 2 * chiSegment**3) / (1 + deltaH)
                    horizontalBendingRotationSway = -(-6 * chiSegment + 6 * chiSegment**2) / (1 + deltaH) / segmentLength
                    horizontalBendingDeflectionYaw = -(deltaH/2 * chiSegment + (2 - deltaH)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + deltaH)
                    horizontalBendingRotationYaw = ((-2 + deltaH) * chiSegment + 3 * chiSegment**2) / (1 + deltaH)

                    surgeDofDisplacements[face,:] = np.array([chiSegment, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([-yOffset * horizontalBendingRotationSway, horizontalBendingDeflectionSway, 0])
                    heaveDofDisplacements[face,:] = np.array([ zOffset * verticalBendingRotationHeave, 0, verticalBendingDeflectionHeave])

                    rollDofDisplacements[face,:]  = np.array([0, -zOffset * chiSegment, yOffset * chiSegment])
                    pitchDofDisplacements[face,:] = np.array([ zOffset * verticalBendingRotationPitch, 0, verticalBendingDeflectionPitch])
                    yawDofDisplacements[face,:]   = np.array([-yOffset * horizontalBendingRotationYaw, horizontalBendingDeflectionYaw, 0])

            # segment after the vertex
            segment = vertex

            if segment == self.numberSegments:
                for face in self.facesOnSegment[segment]:
                    surgeDofDisplacements[face,:] = np.array([1, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([0, 1, 0])
                    heaveDofDisplacements[face,:] = np.array([0, 0, 1])

                    rollDofDisplacements[face,:]  = np.array([0,                        -self.zOffsetFaces[face], self.yOffsetFaces[face]  ])
                    pitchDofDisplacements[face,:] = np.array([self.zOffsetFaces[face],  0,                        -self.xSegmentFaces[face]])
                    yawDofDisplacements[face,:]   = np.array([-self.yOffsetFaces[face], self.xSegmentFaces[face], 0                        ])
            else:
                segmentLength = self.segmentLengths[segment]

                for face in self.facesOnSegment[segment]:
                    chiSegment = self.chiSegmentFaces[face]
                    yOffset = self.yOffsetFaces[face]
                    zOffset = self.zOffsetFaces[face]

                    deltaV = self.verticalShearCorrections[segment]
                    deltaH = self.horizontalShearCorrections[segment]

                    verticalBendingDeflectionHeave = 1 + (-deltaV * chiSegment - 3 * chiSegment**2 + 2 * chiSegment**3) / (1 + deltaV)
                    verticalBendingRotationHeave = (6 * chiSegment - 6 * chiSegment**2) / segmentLength / (1 + deltaV)
                    verticalBendingDeflectionPitch = ((-2 - deltaV)/2 * chiSegment + (4 + deltaV)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + deltaV)
                    verticalBendingRotationPitch = 1 + ((-4 - deltaV) * chiSegment + 3 * chiSegment**2) / (1 + deltaV)

                    horizontalBendingDeflectionSway = 1 + (-deltaH * chiSegment - 3 * chiSegment**2 + 2 * chiSegment**3) / (1 + deltaH)
                    horizontalBendingRotationSway = -(6 * chiSegment - 6 * chiSegment**2) / segmentLength / (1 + deltaH)
                    horizontalBendingDeflectionYaw = -((-2 - deltaH)/2 * chiSegment + (4 + deltaH)/2 * chiSegment**2 - chiSegment**3) * segmentLength / (1 + deltaH)
                    horizontalBendingRotationYaw = 1 + ((-4 - deltaV) * chiSegment + 3 * chiSegment**2) / (1 + deltaV)

                    surgeDofDisplacements[face,:] = np.array([1 - chiSegment, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([-yOffset * horizontalBendingRotationSway, horizontalBendingDeflectionSway, 0])
                    heaveDofDisplacements[face,:] = np.array([ zOffset * verticalBendingRotationHeave, 0, verticalBendingDeflectionHeave])

                    rollDofDisplacements[face,:]  = np.array([0, -zOffset * (1 - chiSegment), yOffset * (1 - chiSegment)])
                    pitchDofDisplacements[face,:] = np.array([ zOffset * verticalBendingRotationPitch, 0, verticalBendingDeflectionPitch])
                    yawDofDisplacements[face,:]   = np.array([-yOffset * horizontalBendingRotationYaw, horizontalBendingDeflectionYaw, 0])

            dofs[surgeDofName] = surgeDofDisplacements
            dofs[swayDofName]  = swayDofDisplacements
            dofs[heaveDofName] = heaveDofDisplacements

            dofs[rollDofName]  = rollDofDisplacements
            dofs[pitchDofName] = pitchDofDisplacements
            dofs[yawDofName]   = yawDofDisplacements

        return dofs


class SpringingResults:
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