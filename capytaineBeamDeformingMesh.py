import sys
import numpy as np
import numpy.linalg as la
import capytaine as cpt
import xarray as xr



class Beam:
    """
    Class used to define the beam that determines the deformations of the mesh and the structural behavior of the problem.
    """
    def __init__(self, initialPoint, length, segments):
        """
        Instantiates a Beam variable according to the parameters provided. The beam is assumed to be
        straight, oriented towards the positive x direction, and divided into segments of equal length.

        :param initialPoint: Array containing the x, y and z coordinates of the beam's initial
            point, in meters, in the coordinate system given by the mesh. The longitudinal coordinate
            within the beam is zero at that point, and increases towards the positive x direction.
        :type initialPoint: (3,)-numpy.ndarray

        :param length: Length of the beam, in meters.
        :type length: float

        :param segments: Number of segments, or finite elements, into which the beam will be divided.
            Note that the number of vertices will be ```segments + 1``.
        :type segments: int
        """
        if length == 0:
            sys.exit('Invalid beam definition: zero length.')

        if segments == 0:
            sys.exit('Invalid beam definition: zero segments.')

        self.initialPoint = initialPoint
        self.length = length
        self.segments = segments

        self.segmentLength = length/segments
        self.vertices = self.segments + 1

        self.normal = np.array([1,0,0])
    
    def UniformlyDistributedMassMatrix(self, linearDensities :np.ndarray):
        """
        Creates a mass matrix for the Finite Elements Analysis, assuming linearly uniformly distributed
        mass within each segment. Throughout, n corresponds to the beam's number of vertices.

        :param linearDensities: Array containing the linear mass density of each segment of the beam,
            in kg/m.
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type linearDensities: (n,)-numpy.ndarray
        
        :returns: Mass matrix for the FEA analysis.
        :rtype: (6*n, 6*n)-numpy.ndarray
        """
        if linearDensities.size != self.segments:
            sys.exit('Invalid linear densities definition: number of linear densities provided does not match number of beam segments.')

        massMatrix = np.zeros([self.vertices * 6, self.vertices * 6])

        for i in range(self.segments):
            segmentMassOver420 = linearDensities[i] * self.segmentLength / 420

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

            massMatrix[z1  , tau1] += 22 * self.segmentLength * segmentMassOver420
            massMatrix[tau1, z1  ] += 22 * self.segmentLength * segmentMassOver420
            massMatrix[z1  , tau2] += -13 * self.segmentLength * segmentMassOver420
            massMatrix[tau2, z1  ] += -13 * self.segmentLength * segmentMassOver420
            massMatrix[tau1, z2  ] += 13 * self.segmentLength * segmentMassOver420
            massMatrix[z2  , tau1] += 13 * self.segmentLength * segmentMassOver420
            massMatrix[z2  , tau2] += -22 * self.segmentLength * segmentMassOver420
            massMatrix[tau2, z2  ] += -22 * self.segmentLength * segmentMassOver420

            massMatrix[tau1, tau1] += 4 * self.segmentLength**2 * segmentMassOver420
            massMatrix[tau1, tau2] += -3 * self.segmentLength**2 * segmentMassOver420
            massMatrix[tau2, tau1] += -3 * self.segmentLength**2 * segmentMassOver420
            massMatrix[tau2, tau2] += 4 * self.segmentLength**2 * segmentMassOver420

            # horizontal bending motion
            massMatrix[y1, y1] += 156 * segmentMassOver420
            massMatrix[y1, y2] += 54 * segmentMassOver420
            massMatrix[y2, y1] += 54 * segmentMassOver420
            massMatrix[y2, y2] += 156 * segmentMassOver420

            massMatrix[y1  , psi1] += 22 * self.segmentLength * segmentMassOver420
            massMatrix[psi1, y1  ] += 22 * self.segmentLength * segmentMassOver420
            massMatrix[y1  , psi2] += -13 * self.segmentLength * segmentMassOver420
            massMatrix[psi2, y1  ] += -13 * self.segmentLength * segmentMassOver420
            massMatrix[psi1, y2  ] += 13 * self.segmentLength * segmentMassOver420
            massMatrix[y2  , psi1] += 13 * self.segmentLength * segmentMassOver420
            massMatrix[y2  , psi2] += -22 * self.segmentLength * segmentMassOver420
            massMatrix[psi2, y2  ] += -22 * self.segmentLength * segmentMassOver420

            massMatrix[psi1, psi1] += 4 * self.segmentLength**2 * segmentMassOver420
            massMatrix[psi1, psi2] += -3 * self.segmentLength**2 * segmentMassOver420
            massMatrix[psi2, psi1] += -3 * self.segmentLength**2 * segmentMassOver420
            massMatrix[psi2, psi2] += 4 * self.segmentLength**2 * segmentMassOver420

            # torsional motion: TODO
            massMatrix[phi1, phi1] = 2 * segmentMassOver420
            massMatrix[phi1, phi2] = 1 * segmentMassOver420
            massMatrix[phi2, phi1] = 1 * segmentMassOver420
            massMatrix[phi2, phi2] = 2 * segmentMassOver420
        
        return massMatrix

    def StiffnessMatrix(self, crossSectionAreas :np.ndarray, verticalAreaMoments :np.ndarray, horizontalAreaMoments :np.ndarray, verticalTimoshenkoCoefs :np.ndarray, horizontalTimoshenkoCoefs :np.ndarray, youngsModulus, shearModulus):
        """
        Creates a stiffness matrix for the Finite Elements Analysis according to Timoshenko beam theory.
        Throughout, n corresponds to the beam's number of vertices.

        :param crossSectionAreas: Array containing the cross sectional areas of each segment of the beam,
            in m^2.
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type crossSectionAreas: (n,)-numpy.ndarray

        :param verticalAreaMoments: Array containing the vertical area moment of the cross section for each
            segment of the beam, in m^4.
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type verticalAreaMoments: (n,)-numpy.ndarray

        :param horizontalAreaMoments: Array containing the horizontal area moment of the cross section for each
            segment of the beam, in m^4.
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type horizontalAreaMoments: (n,)-numpy.ndarray

        :param verticalTimoshenkoCoefs: Array containing the vertical Timoshenko coefficient of the cross section
            for each segment of the beam. The shear correction for a Timoshenko beam is calculated as

            ``verticalShearCorrection = (12 * youngsModulus * areaMoment) / (verticalTimoshenkoCoefs * crossSectionArea * shearModulus * segmentLength**2)``
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type verticalTimoshenkoCoefs: (n,)-numpy.ndarray

        :param horizontalTimoshenkoCoefs: Array containing the horizontal Timoshenko coefficient of the cross
            section for each segment of the beam. The shear correction for a Timoshenko beam is calculated as

            ``horizontalShearCorrection = (12 * youngsModulus * areaMoment) / (horizontalTimoshenkoCoefs * crossSectionArea * shearModulus * segmentLength**2)``
            
            The ``0``-th element corresponds to the segment closest to the beam's initial point.
            The following elements are ordered in ascending x coordinate.
        :type horizontalTimoshenkoCoefs: (n,)-numpy.ndarray

        :param youngsModulus: Young's modulus for the structural material, in Pa.
        :type youngsModulus: float

        :param shearModulus: Shear modulus for the structural material, in Pa.
        :type shearModulus: float
        
        :returns: Stiffness matrix for the FEA analysis.
        :rtype: (6*n, 6*n)-numpy.ndarray
        """
        if crossSectionAreas.size != self.segments:
            sys.exit('Invalid cross sectional areas definition: number of areas provided does not match number of beam segments.')

        if verticalAreaMoments.size != self.segments:
            sys.exit('Invalid vertical area moments definition: number of area moments provided does not match number of beam segments.')

        if horizontalAreaMoments.size != self.segments:
            sys.exit('Invalid horizontal area moments definition: number of area moments provided does not match number of beam segments.')
        
        if verticalTimoshenkoCoefs.size != self.segments:
            sys.exit('Invalid vertical Timoshenko coefficients definition: number of Timoshenko coefficients provided does not match number of beam segments.')

        if horizontalTimoshenkoCoefs.size != self.segments:
            sys.exit('Invalid horizontal Timoshenko coefficients definition: number of Timoshenko coefficients provided does not match number of beam segments.')

        stiffnessMatrix = np.zeros([self.vertices * 6, self.vertices * 6])

        verticalShearAreas = verticalTimoshenkoCoefs * crossSectionAreas
        horizontalShearAreas = horizontalTimoshenkoCoefs * crossSectionAreas

        verticalShearCorrections = 12 * youngsModulus * verticalAreaMoments / (verticalShearAreas * shearModulus * self.segmentLength**2)
        horizontalShearCorrections = 12 * youngsModulus * horizontalAreaMoments / (horizontalShearAreas * shearModulus * self.segmentLength**2)

        for i in range(self.segments):
            shearCorrectionVertical = verticalShearCorrections[i]
            shearCorrectionHorizontal = horizontalShearCorrections[i]

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
            EAOverL = youngsModulus * crossSectionAreas[i] / self.segmentLength

            stiffnessMatrix[x1, x1] += EAOverL
            stiffnessMatrix[x1, x2] += -EAOverL
            stiffnessMatrix[x2, x1] += -EAOverL
            stiffnessMatrix[x2, x2] += EAOverL

            # vertical bending
            EIOverCorrection = youngsModulus * verticalAreaMoments[i] / (1 + shearCorrectionVertical)

            stiffnessMatrix[z1, z1] += 12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[z1, z2] += -12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[z2, z1] += -12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[z2, z2] += 12 * EIOverCorrection / self.segmentLength**3

            stiffnessMatrix[z1  , tau1] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[tau1, z1  ] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[z1  , tau2] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[tau2, z1  ] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[tau1, z2  ] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[z2  , tau1] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[z2  , tau2] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[tau2, z2  ] += -6 * EIOverCorrection / self.segmentLength**2

            stiffnessMatrix[tau1, tau1] += (4 + shearCorrectionVertical) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[tau1, tau2] += (2 - shearCorrectionVertical) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[tau2, tau1] += (2 - shearCorrectionVertical) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[tau2, tau2] += (4 + shearCorrectionVertical) * EIOverCorrection / self.segmentLength

            # horizontal bending
            EIOverCorrection = youngsModulus * horizontalAreaMoments[i] / (1 + shearCorrectionHorizontal)

            stiffnessMatrix[y1, y1] += 12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[y1, y2] += -12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[y2, y1] += -12 * EIOverCorrection / self.segmentLength**3
            stiffnessMatrix[y2, y2] += 12 * EIOverCorrection / self.segmentLength**3

            stiffnessMatrix[y1  , psi1] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[psi1, y1  ] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[y1  , psi2] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[psi2, y1  ] += 6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[psi1, y2  ] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[y2  , psi1] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[y2  , psi2] += -6 * EIOverCorrection / self.segmentLength**2
            stiffnessMatrix[psi2, y2  ] += -6 * EIOverCorrection / self.segmentLength**2

            stiffnessMatrix[psi1, psi1] += (4 + shearCorrectionHorizontal) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[psi1, psi2] += (2 - shearCorrectionHorizontal) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[psi2, psi1] += (2 - shearCorrectionHorizontal) * EIOverCorrection / self.segmentLength
            stiffnessMatrix[psi2, psi2] += (4 + shearCorrectionHorizontal) * EIOverCorrection / self.segmentLength

            stiffnessMatrix[phi1, phi1] = EAOverL
            stiffnessMatrix[phi1, phi2] = -EAOverL
            stiffnessMatrix[phi2, phi1] = -EAOverL
            stiffnessMatrix[phi2, phi2] = EAOverL

        self.crossSectionAreas = crossSectionAreas
        self.verticalAreaMoments = verticalAreaMoments
        self.horizontalAreaMoments = horizontalAreaMoments
        self.verticalShearCorrections = verticalShearCorrections
        self.horizontalShearCorrections = horizontalShearCorrections
        self.youngsModulus = youngsModulus
        self.shearModulus = shearModulus
        
        return stiffnessMatrix

    def SegmentBendingMoment(self, xSegment, segmentDisplacements, areaMoment, shearCorrection, plane):
        """
        Calculates the vertical or horizontal bending moments at a one or more points within a beam segment
        given their x coordinates within the segment and the displacements of the vertices around it.

        :param xSegment: Value or array of values of the x coordinate within the beam segment, in meters, of
            the point or points at which the bending moment is to be calculated.
        :type xSegment: float or (,m)-numpy.ndarray

        :param segmentDisplacements: Array of displacements, in meters and radians, of the beam vertices before
            and after the beam segment at which the bending moment is to be calculated.
        :type segmentDisplacements: (,12)-numpy.ndarray

        :param areaMoment: Vertical or horizontal geometric area moment of the beam section of the segment of interest,
            in m^4.
        :type areaMoment: float

        :param shearCorrection: Vertical or horizontal shear correction of the segment of interest.
        :type shearCorrection: float

        :param plane: String identifying if the vertical, ``'v'``, or horizontal, ``'h'``, bending moment is to
            be calculated.
        :type plane: str

        :returns: Value or array of values of the vertical or horizontal bending moment at the point or points
            indicated by ``xSegment``, in Nm.
        :rtype: float or (,m)-numpy.ndarray
        """
        if type(xSegment) == np.ndarray:
            if np.min(xSegment) < 0 or np.max(xSegment) > self.segmentLength:
                sys.exit('All values for the x coordinate within the beam segment must be between 0 and the segment length.')
        else:
            if xSegment < 0 or xSegment > self.segmentLength:
                sys.exit('The value for the x coordinate within the beam segment must be between 0 and the segment length.')

        if plane == 'v':
            linear1 = segmentDisplacements[2]
            angular1 = segmentDisplacements[4]
            linear2 = segmentDisplacements[8]
            angular2 = segmentDisplacements[10]
        elif plane == 'h':
            linear1 = segmentDisplacements[1]
            angular1 = segmentDisplacements[5]
            linear2 = segmentDisplacements[7]
            angular2 = segmentDisplacements[11]
        else:
            sys.exit('Bending moment must be vertical (v) or horizontal (h).')

        L = self.segmentLength

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
        n corresponds to the beam's number of vertices.

        :param x: Value or array of values of the x coordinate within the beam, in meters, of the point
            or points at which the bending moment is to be calculated.
        :type x: float or (,m)-numpy.ndarray

        :param displacements: Array of displacements, in meters and radians, of all of the beam vertices.
        :type displacements: (,6*n)-numpy.ndarray

        :param plane: String identifying if the vertical, ``'v'``, or horizontal, ``'h'``, bending moment is to
            be calculated.
        :type plane: str, optional. Default: ``'v'``.

        :returns: Value or array of values of the vertical or horizontal bending moment at the point or points
            indicated by ``x``, in Nm.
        :rtype: float or (,m)-numpy.ndarray
        """
        if not (plane == 'v' or plane == 'h'):
            sys.exit('Bending moment must be vertical (v) or horizontal (h).')

        if not self.vertices*6 == displacements.size:
            sys.exit('Wrong size of displacements vector')

        if type(x) == np.ndarray:
            if not np.sort(x) == x:
                sys.exit('The array of x positions at which the bending moment will be calculated must be sorted.')

            if x[0] < 0 or x[-1] > self.length:
                sys.exit('All positions x at which the bending moment will be calculated must be between 0 and the beam length.')
        else:
            if x < 0 or x > self.length:
                sys.exit('The position x at which the bending moment will be calculated must be between 0 and the beam length.')

            leftVertexIndex = int(np.floor(x/self.segmentLength))
            coordinateSegment = x - self.segmentLength * leftVertexIndex

            segmentDisplacements = displacements[6*leftVertexIndex : 6*(leftVertexIndex + 2)]

            if plane == 'v':
                verticalAreaMoment = self.verticalAreaMoments[leftVertexIndex]
                verticalShearCorrection = self.verticalShearCorrections[leftVertexIndex]
                return self.SegmentBendingMoment(coordinateSegment, segmentDisplacements, verticalAreaMoment, verticalShearCorrection, 'v')
            else:
                horizontalAreaMoment = self.horizontalAreaMoments[leftVertexIndex]
                horizontalShearCorrection = self.horizontalShearCorrections[leftVertexIndex]
                return self.SegmentBendingMoment(coordinateSegment, segmentDisplacements, horizontalAreaMoment, horizontalShearCorrection, 'h')

        bendingMomentDistribution = np.zeros([x.size], dtype = np.complex128)

        for i in range(self.segments):
            if i == self.segments - 1:
                mask = (x >= i*self.segmentLength) & (x <= self.length)
            else:
                mask = (x >= i*self.segmentLength) & (x < (i+1)*self.segmentLength)

            coordinatesSegment = x[mask] - i*self.segmentLength

            if not coordinatesSegment.size == 0:
                segmentDisplacements = displacements[6*i : 6*(i + 2)]

                if plane == 'v':
                    verticalAreaMoment = self.verticalAreaMoments[i]
                    verticalShearCorrection = self.verticalShearCorrections[i]
                    bendingMomentDistribution[mask] = self.SegmentBendingMoment(coordinatesSegment, segmentDisplacements, verticalAreaMoment, verticalShearCorrection, 'v')
                else:
                    horizontalAreaMoment = self.horizontalAreaMoments[i]
                    horizontalShearCorrection = self.horizontalShearCorrections[i]
                    bendingMomentDistribution[mask] = self.SegmentBendingMoment(coordinatesSegment, segmentDisplacements, horizontalAreaMoment, horizontalShearCorrection, 'h')

        return bendingMomentDistribution


class MeshBeamProperties:
    """
    Class used to manage the interaction between the beam model and the hull mesh. It allows to calculate
    the Capytaine degrees of freedom associated to the beam's vertices' movements.
    """
    def __init__(self, hullMesh : cpt.Mesh, beam : Beam):
        """
        Instantiates a MeshBeamProperties variable and stores in it information about the interaction
        between the beam model and the mesh. In particular, it calculates the beam segment to which each
        mesh face is associated, as well as their corresponding beam coordinates.

        :param hullMesh: Mesh of the ship's hull.
        :type hullMesh: capytaine.Mesh

        :param beam: Beam modelling the hull girder.
        :type beam: Beam

        :returns: Class object containing the following attributes.

            * numberMeshFaces (int): Number of faces of ``hullMesh``.
            * beamSegments (int): Number of segments in ``beam``.
            * beamVertices (int): Number of vertices in ``beam``.
            * beamSegmentLength (float): Length of a each segment of ``beam``, in meters.
            * segmentOfFace (numpy.ndarray): Array whose ``i``-th component contains the beam segment that
                corresponds to the ``i``-th face of the mesh. Faces with x coordinate smaller than the
                beam's first vertex are associated to segment ``-1``, and faces with x coordinate greater
                than the beam's last vertex are associated to segment ``beamSegments``. The rest of faces
                are associated to a segment running from ``0`` to ``beamSegments-1``.
            * xSegmentFaces (numpy.ndarray): Array whose ``i``-th component contains the value of the x
                coordinate within its segment of the ``i``-th face, in meters. For segments within the beam,
                this value ranges between ``0`` and the length of the beam's segments. For segment ``0``,
                before the first vertex, it is a negative value, as that segment's coordinate origin is
                placed at the first vertex. For segment ``beamVertices``, after the last vertex, it is a
                positive value, as its coordinate origin is placed at the last vertex.
            * chiSegmentFaces (numpy.ndarray): Array whose ``i``-th component contains the value of the
                non-dimensionalized x coordinate within the segment of the ``i``-th face. This array
                is simply ``xSegmentFaces / beamSegmentLength``.
            * yOffsetFaces (numpy.ndarray): Array whose ``i``-th component contains the value of the
                y coordinate, in meters, of face ``i`` with respect to the beam's neutral axis.
            * zOffsetFaces (numpy.ndarray): Array whose ``i``-th component contains the value of the
                z coordinate, in meters, of face ``i`` with respect to the beam's neutral axis.
        :rtype: MeshBeamProperties
        """
        self.numberMeshFaces = hullMesh.nb_faces

        self.beamSegments = beam.segments
        self.beamVertices = beam.vertices
        self.beamSegmentLength = beam.segmentLength
        self.verticalShearCorrections = beam.verticalShearCorrections
        self.horizontalShearCorrections = beam.horizontalShearCorrections

        self.segmentOfFace = np.zeros([self.numberMeshFaces], dtype=int)

        for i in range(self.numberMeshFaces):
            normalizedLongitudinalPositionInBeam = np.dot(hullMesh.faces_centers[i] - beam.initialPoint, beam.normal)/beam.length

            if normalizedLongitudinalPositionInBeam < 0:
                self.segmentOfFace[i] = -1
            elif normalizedLongitudinalPositionInBeam > 1:
                self.segmentOfFace[i] = beam.segments
            else:
                self.segmentOfFace[i] = np.floor(normalizedLongitudinalPositionInBeam * beam.segments)


        self.facesOnSegment = np.ndarray([beam.segments + 2], dtype=object)

        for i in range(-1, beam.segments + 1):
            self.facesOnSegment[i] = np.where(self.segmentOfFace == i)[0]


        self.xSegmentFaces = np.zeros([self.numberMeshFaces])
        self.chiSegmentFaces = np.zeros([self.numberMeshFaces])
        self.yOffsetFaces = np.zeros([self.numberMeshFaces])
        self.zOffsetFaces = np.zeros([self.numberMeshFaces])

        # Segment -1: before the start of the beam
        xInitialSegment = beam.initialPoint[0]

        for face in self.facesOnSegment[-1]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = -1

            self.yOffsetFaces[face] = hullMesh.faces_centers[face,1] - beam.initialPoint[1]
            self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - beam.initialPoint[2]

        # Segments 0 to beam.segments - 1: within the beam
        for segment in range(0, beam.segments):
            xInitialSegment = beam.initialPoint[0] + beam.segmentLength * segment

            for face in self.facesOnSegment[segment]:
                self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
                self.chiSegmentFaces[face] = self.xSegmentFaces[face] / beam.segmentLength

                self.yOffsetFaces[face] = hullMesh.faces_centers[face,1] - beam.initialPoint[1]
                self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - beam.initialPoint[2]

        # Segment beam.segments: after the end of the beam
        xInitialSegment = beam.initialPoint[0] + beam.length

        for face in self.facesOnSegment[beam.segments]:
            self.xSegmentFaces[face] = hullMesh.faces_centers[face,0] - xInitialSegment
            self.chiSegmentFaces[face] = 2

            self.yOffsetFaces[face] = hullMesh.faces_centers[face,1] - beam.initialPoint[1]
            self.zOffsetFaces[face] = hullMesh.faces_centers[face,2] - beam.initialPoint[2]

    def CreateDOFs(self):
        """
        Creates the degrees of freedom corresponding to the object's mesh and beam.

        This function takes the object's mesh and beam and creates Capytaine degrees of freedom.
        To each beam vertex, six degrees of freedom are associated: three linear and three angular,
        according to the right-hand rule sign convention. The degrees of freedom corresponding to
        the ``i``-th vertex are labeled by the following strings.

        Surge ``'x%d'%i``
        
        Sway ``'y%d'%i``
        
        Heave ``'z%d'%i``
        
        Roll ``'phi%d'%i``
        
        Pitch ``'tau%d'%i``
        
        Yaw ``'psi%d'%i``

        The beam's neutral axis is deformed between vertices according to the FEM interpolation
        polynomials for beam theory. That is, cubic for bending, linear for axial and torsional.
        This extends to a deformation of the whole mesh under the assumption that sections remain
        perpendicular to the neutral axis. The field of displacements is linearized, as required by
        hydrodynamic panel code formulations. The mesh regions before the first and after the last
        beam vertices are deformed as rigid bodies fixed to the endpoint vertex.

        :returns: A dictionary containing the degrees of freedom, ready for input to Capytaine.
        :rtype: dict
        """
        dofs = {}

        for vertex in range(self.beamVertices):
            surgeDofName = 'x%d'%vertex # axial
            swayDofName  = 'y%d'%vertex # bending horizontal
            heaveDofName = 'z%d'%vertex # bending vertical

            rollDofName  = 'roll%d'%vertex  # torsion
            pitchDofName = 'pitch%d'%vertex # bending horizontal
            yawDofName   = 'yaw%d'%vertex   # bending vertical

            surgeDofDisplacements = np.zeros([self.numberMeshFaces, 3])
            swayDofDisplacements  = np.zeros([self.numberMeshFaces, 3])
            heaveDofDisplacements = np.zeros([self.numberMeshFaces, 3])

            rollDofDisplacements  = np.zeros([self.numberMeshFaces, 3])
            pitchDofDisplacements = np.zeros([self.numberMeshFaces, 3])
            yawDofDisplacements   = np.zeros([self.numberMeshFaces, 3])

            segmentLength = self.beamSegmentLength

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

            if segment == self.beamSegments:
                for face in self.facesOnSegment[segment]:
                    surgeDofDisplacements[face,:] = np.array([1, 0, 0])
                    swayDofDisplacements[face,:]  = np.array([0, 1, 0])
                    heaveDofDisplacements[face,:] = np.array([0, 0, 1])

                    rollDofDisplacements[face,:]  = np.array([0,                        -self.zOffsetFaces[face], self.yOffsetFaces[face]  ])
                    pitchDofDisplacements[face,:] = np.array([self.zOffsetFaces[face],  0,                        -self.xSegmentFaces[face]])
                    yawDofDisplacements[face,:]   = np.array([-self.yOffsetFaces[face], self.xSegmentFaces[face], 0                        ])
            else:
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
        of vertices.

        :param massMatrix: Structural mass matrix of the ship, as a Finite Elements Analysis mass
            matrix for the hull girder beam.
        :type massMatrix: (6*n,6*n)-numpy.ndarray

        :param stiffnessMatrix: Structural stiffness matrix of the ship, as a Finite Elements
            Analysis stiffness matrix for the hull girder beam.
        :type stiffnessMatrix: (6*n,6*n)-numpy.ndarray

        :param hydrostaticStiffness: Capytaine hydrostatic stiffness results for the FloatingBody
            defined by the ship, as returned by the ``compute_hydrostatic_stiffness`` method of
            the ``capytaine.FloatingBody`` class. The FloatingBody must include the degrees of
            freedom given by the beam vertex motions, as calculated by the ``CreateDOFs`` method
            of the ``MeshBeamProperties`` class.
        :type hydrostaticStiffness: xarray.DataArray

        :param hydrodynamicResults: Dataset of Capytaine linear potential flow results for the
            FloatingBody defined by the ship on a test matrix with different wave frequencies,
            directions and water depths, as returned by the ``fill_dataset`` method of the
            ``capytaine.BEMSolver`` class. The FloatingBody must include the degrees of freedom
            given by the beam vertex motions, as calculated by the ``MeshBeamProperties.CreateDOFs``
            method.
        :type hydrodynamicResults: xarray.Dataset

        :returns: Class object containing the following attributes.

            * displacementAmplitudes (xarray.DataArray): Array of results for the complex amplitudes
                of the springing motions of the beam vertices divided by wave height, in m/m and
                rad/m. The motions are sinusoidal, with the real part of the amplitude being the
                displacement at t = 0 and the imaginary part being the displacement, with
                opposite sign, after one fourth of the period. The array's dimensions are labeled
                ``'omega'``, ``'wave_direction'``, ``'water_depth'`` and ``'dof'``. The length and
                coordinates associated to each of the first three dimensions match those of the
                ``added_mass`` and ``radiation_damping`` attributes of ``hydrodynamicResults``, and
                are determined by the test matrix passed to the Capytaine BEM Solver when calculating
                these results. The ``'dof'`` dimension has a length of 6n and indexes the degree of
                freedom each amplitude corresponds to.
            * massMatrix (xarray.DataArray): The provided ``massMatrix`` as an ``xarray.DataArray``,
                with dimensions labeled ``'influenced_dof'`` and ``'radiating_dof'``.
            * stiffnessMatrix (xarray.DataArray): The provided ``stiffnessMatrix`` as an ``xarray.DataArray``,
                with dimensions labeled ``'influenced_dof'`` and ``'radiating_dof'``.
        :rtype: SpringingResults
        """
        self.massMatrix = xr.DataArray(massMatrix, dims = ['influenced_dof', 'radiating_dof'])
        self.stiffnessMatrix = xr.DataArray(stiffnessMatrix, dims = ['influenced_dof', 'radiating_dof'])

        self.forcesFromAmplitudesMatrices = - (hydrodynamicResults.added_mass + self.massMatrix) * hydrodynamicResults.omega**2 + complex(0,1) * hydrodynamicResults.omega * hydrodynamicResults.radiation_damping + (self.stiffnessMatrix + hydrostaticStiffness)

        self.amplitudesFromForcesMatrices = xr.DataArray(la.inv(self.forcesFromAmplitudesMatrices), dims = ['omega', 'radiating_dof', 'influenced_dof'])

        self.displacementAmplitudes = xr.dot(hydrodynamicResults.excitation_force, self.amplitudesFromForcesMatrices, dims = ['influenced_dof'])
        self.displacementAmplitudes = self.displacementAmplitudes.rename({'radiating_dof': 'dof'})

        self.hydrostaticStiffness = hydrostaticStiffness
        self.hydrodynamicResults = hydrodynamicResults