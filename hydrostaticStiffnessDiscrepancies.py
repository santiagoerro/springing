import numpy as np
import capytaine as cpt
import springing as spr

np.set_printoptions(linewidth=100000)

hullLength = 1
hullBreadth = 1
hullDraft = 1



hullSize = (hullLength, hullBreadth, hullDraft)
hullCenter = (0, 0, 0)
meshResolution = (10, 10, 5)

hullMesh = cpt.mesh_parallelepiped(size = hullSize, center = hullCenter, name = 'hull', resolution = meshResolution).immersed_part()

rigidBodyDofs = cpt.rigid_body_dofs(hullCenter)
rigidBody = cpt.FloatingBody(mesh = hullMesh, dofs = rigidBodyDofs, center_of_mass = hullCenter)
rigidBodyStiffnessMatrix = rigidBody.compute_hydrostatic_stiffness()

myDofs = {}
myDofs['x'] = np.zeros([hullMesh.nb_faces, 3])
myDofs['x'][:, 0] = 1
myDofs['p'] = np.zeros([hullMesh.nb_faces, 3])
myDofs['p'][:, 2] = hullMesh.faces_centers[:, 0]

myBody = cpt.FloatingBody(mesh = hullMesh, dofs = myDofs, center_of_mass = hullCenter)
myBodyStiffnessMatrix = myBody.compute_hydrostatic_stiffness()
myMethodStiffnessMatrix = spr.ComputeHydrostaticStiffness(myBody, 1000, 9.81)



print(rigidBodyStiffnessMatrix)
print(myBodyStiffnessMatrix)
print(myMethodStiffnessMatrix)