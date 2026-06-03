import numpy as np
import capytaine as cpt
import springing as spr

np.set_printoptions(linewidth=100000)

hullLength = 1
hullBreadth = 1
hullDraft = 0.5



hullSize = (hullLength, hullBreadth, hullDraft)
hullCenter = (0, 0, 0)
meshResolution = (10, 10, 5)

hullMesh: cpt.Mesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()

rigidBodyDofs = cpt.rigid_body_dofs(hullCenter)
rigidBody = cpt.FloatingBody(mesh = hullMesh, dofs = rigidBodyDofs, center_of_mass = hullCenter)
rigidBodyStiffnessMatrix = rigidBody.compute_hydrostatic_stiffness()

# myDofs = {}
# myDofs['x'] = np.zeros([hullMesh.nb_faces, 3])
# myDofs['x'][:, 0] = 1
# myDofs['p'] = np.zeros([hullMesh.nb_faces, 3])
# myDofs['p'][:, 2] = -hullMesh.faces_centers[:, 0]
# myDofs['p'][:, 0] = hullMesh.faces_centers[:, 2]

freeSurfaceVertices = np.isclose(hullMesh.vertices[:, 2], 0)

vertexDofs = {}
vertexDofs['x'] = np.zeros([hullMesh.nb_vertices, 3])
vertexDofs['x'][:, 0] = 1
vertexDofs['x'][freeSurfaceVertices, 2] = 0
vertexDofs['p'] = np.zeros([hullMesh.nb_vertices, 3])
vertexDofs['p'][:, 2] = -hullMesh.vertices[:, 0]
vertexDofs['p'][:, 0] = hullMesh.vertices[:, 2]
vertexDofs['p'][freeSurfaceVertices, 2] = 0

myDofs = {}
myDofs['x'] = np.mean(vertexDofs['x'][hullMesh.faces, :], axis = 1)
myDofs['p'] = np.mean(vertexDofs['p'][hullMesh.faces, :], axis = 1)

dofJacobians = {}
dofJacobians['x'] = np.zeros([hullMesh.nb_faces, 3, 3])
dofJacobians['p'] = np.zeros([hullMesh.nb_faces, 3, 3])
dofJacobians['p'][:, 0, 2] = 1
dofJacobians['p'][:, 2, 0] = -1

myBody = cpt.FloatingBody(mesh = hullMesh, dofs = myDofs, center_of_mass = hullCenter)
myBodyStiffnessMatrix = myBody.compute_hydrostatic_stiffness()
myFunctionStiffnessMatrix = spr.ComputeHydrostaticStiffness(myBody, 1000, 9.81)
myFunctionMyMethodStiffnessMatrix = spr.ComputeHydrostaticStiffnessNewMethod(myBody, vertexDofs, dofJacobians, 1000, 9.81)



print(rigidBodyStiffnessMatrix)
print(myBodyStiffnessMatrix)
print(myFunctionStiffnessMatrix)
print(myFunctionMyMethodStiffnessMatrix)