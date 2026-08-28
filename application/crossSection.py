from sectionproperties.analysis import Section
from sectionproperties.pre import Geometry



geom = Geometry.from_dxf(dxf_filepath="section.dxf")
geom.create_mesh(mesh_sizes=[0.5])
sec = Section(geometry=geom)

sec.calculate_geometric_properties()
sec.calculate_warping_properties(solver_type="direct")

crossSectionalArea = sec.get_area()

centroid = sec.get_c()
shearCenter = sec.get_sc()

verticalSecondMomentArea = sec.get_ip()[1]
horizontalSecondMomentArea = sec.get_ip()[0]

shearAreaVertical = sec.get_as()[1]
shearAreaHorizontal = sec.get_as()[0]

torsionConstant = sec.get_j()
warpingConstant = sec.get_gamma()



print(f"Cross sectional area: {crossSectionalArea:.3e} mm2")
print()
print(f"Neutral axis over section base: {centroid[1]:.1f} mm")
print(f"Shear center over section base: {shearCenter[1]:.1f} mm")
print()
print(f"Vertical second moment of area:   {verticalSecondMomentArea:.3e} mm4")
print(f"Horizontal second moment of area: {horizontalSecondMomentArea:.3e} mm4")
print()
print(f"Vertical timoshenko coef:   {shearAreaVertical/crossSectionalArea:.3f}")
print(f"Horizontal timoshenko coef: {shearAreaHorizontal/crossSectionalArea:.3f}")
print()
print(f"Torsion constant: {torsionConstant:.3e} mm4")
print(f"Warping constant: {warpingConstant:.3e} mm6")

sec.plot_centroids()