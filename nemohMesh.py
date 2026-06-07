import capytaine as cpt
import os

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

# mesh resolution
panelsPerMeter = 65



# CALCULATIONS

hullDraft = hullDisplacement / (waterDensity * hullLength * hullBreadth)

# mesh generation
panelsLength = int(round(panelsPerMeter * hullLength))
panelsBreadth = int(round(panelsPerMeter * hullBreadth))
panelsDepth = int(round(panelsPerMeter * hullDepth))

hullSize = (hullLength, hullBreadth, hullDepth)
hullCenter = (0, 0, -hullDraft + hullDepth/2)
meshResolution = (panelsLength, panelsBreadth, panelsDepth)

hullMesh: cpt.Mesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()



# OUTPUT

if not os.path.exists('meshes'):
    os.makedirs('meshes')

file = open('meshes/nemohBargeMesh.dat', 'w')
file.write('2 1\n')
for i in range(hullMesh.vertices.shape[0]):
    file.write('%d %e %e %e\n'%(i + 1, hullMesh.vertices[i, 0], hullMesh.vertices[i, 1], hullMesh.vertices[i, 2]))
file.write('0 0 0 0\n')
for i in range(hullMesh.faces.shape[0]):
    file.write('%d %d %d %d\n'%(hullMesh.faces[i, 0] + 1, hullMesh.faces[i, 1] + 1, hullMesh.faces[i, 2] + 1, hullMesh.faces[i, 3] + 1))
file.write('0 0 0 0\n')

file.close()