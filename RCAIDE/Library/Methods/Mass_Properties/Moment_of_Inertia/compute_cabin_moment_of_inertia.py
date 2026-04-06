# RCAIDE/Library/Methods/Mass_Properties/Moment_of_Inertia/compute_cabin_moment_of_inertia.py 
# 
# Created:  Dec 2025, M. Clarke  
 
# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE imports 
from RCAIDE.Library.Methods.Geometry.LOPA.compute_layout_of_passenger_accommodations import compute_layout_of_passenger_accommodations

# python imports
import RNUMPY as rp  
# ----------------------------------------------------------------------------------------------------------------------
#  Compute Cabin Moment of Inertia
# ----------------------------------------------------------------------------------------------------------------------   
def compute_cabin_moment_of_inertia(cabin,fuselage,center_of_gravity = rp.array([[0,0,0]])):  
    """
    Computes the moment of inertia tensor for the cabin.
    
    Parameters
    ----------
    center_of_gravity : list, optional
        Reference point coordinates for moment calculation, defaults to [[0, 0, 0]]
    
    Returns
    -------
    I : ndarray
        3x3 moment of inertia tensor in kg*m^2
    
    See Also
    --------
    RCAIDE.Library.Methods.weights.vehicle.moments_of_inertia.compute_fuselage_moment_of_inertia
        Implementation of the moment of inertia calculation
    """

    # Gather x,y points of the cabin
    if fuselage.layout_of_passenger_accommodations == None:
        compute_layout_of_passenger_accommodations(fuselage)
    half_coords = fuselage.layout_of_passenger_accommodations.cabin_area_coordinates # one side of the x,y coordiantes of the cabin
    coordinates = rp.vstack([rp.hstack([half_coords[:,0], half_coords[::-1,0]]), rp.hstack([half_coords[:,1], -1*half_coords[::-1,1]])]) # Full coordinates

    # Create 3D mesh of the cabin area
    L       = max(cabin.height,0.1) # cabin height with arbitrarily small value to avoid 0 thickness error
    pts1    = rp.column_stack((coordinates[0], coordinates[1], rp.zeros(len(coordinates[0]))))   # z = 0 top surface points
    pts2    = rp.column_stack((coordinates[0], coordinates[1], rp.full((len(coordinates[0]),), L))) # z = L bottom surface points
    all_pts = rp.vstack([pts1, pts2]) # Combine all points for the 3D geometry

    # Rotate points to match the RCAIDE aircraft axes convention (90 deg around X)
    Rot          = rp.scipy.spatial.transform.Rotation.from_euler('x', 90, degrees=True)
    rotated_pts  = Rot.apply(all_pts)

    # Create Convex hull → watertight volume mesh
    hull     = rp.scipy.spatial.ConvexHull(rotated_pts)
    vertices = hull.points
    faces    = hull.simplices

    # Get vertices for each face
    P1 = vertices[faces[:, 0]]
    P2 = vertices[faces[:, 1]]
    P3 = vertices[faces[:, 2]]

    # Compute face volumes (signed tetrahedra volumes from origin)
    V_f    = rp.sum(P1 * rp.cross(P2, P3), axis=1) / 6.0
    volume = rp.sum(V_f)

    # Compute center of mass (centroid)
    S        = P1 + P2 + P3
    centroid = rp.sum(V_f[:, None] * (S / 4.0), axis=0) / volume

    # Compute inertia integrals at origin
    Jxx_int = rp.sum(V_f / 20.0 * (P1[:, 0]**2 + P2[:, 0]**2 + P3[:, 0]**2 + S[:, 0]**2))
    Jyy_int = rp.sum(V_f / 20.0 * (P1[:, 1]**2 + P2[:, 1]**2 + P3[:, 1]**2 + S[:, 1]**2))
    Jzz_int = rp.sum(V_f / 20.0 * (P1[:, 2]**2 + P2[:, 2]**2 + P3[:, 2]**2 + S[:, 2]**2))
    Jxy_int = rp.sum(V_f / 20.0 * (P1[:, 0]*P1[:, 1] + P2[:, 0]*P2[:, 1] + P3[:, 0]*P3[:, 1] + S[:, 0]*S[:, 1]))
    Jxz_int = rp.sum(V_f / 20.0 * (P1[:, 0]*P1[:, 2] + P2[:, 0]*P2[:, 2] + P3[:, 0]*P3[:, 2] + S[:, 0]*S[:, 2]))
    Jyz_int = rp.sum(V_f / 20.0 * (P1[:, 1]*P1[:, 2] + P2[:, 1]*P2[:, 2] + P3[:, 1]*P3[:, 2] + S[:, 1]*S[:, 2]))

    # Moments of inertia tensor at origin
    mass = cabin.mass_properties.mass
    rho  = mass / volume
    I_origin = rp.array([
        [Jyy_int + Jzz_int, -Jxy_int,          -Jxz_int],
        [-Jxy_int,          Jxx_int + Jzz_int, -Jyz_int],
        [-Jxz_int,          -Jyz_int,          Jxx_int + Jyy_int]
    ]) * rho

    # Convert to centroid (reverse parallel axis theorem)
    c          = centroid
    I_centroid = I_origin - mass * (rp.dot(c, c) * rp.identity(3) - rp.outer(c, c))
    
    # Calculate MOI of the cabin relative to the given center of gravity
    I = I_centroid
    
    # additional MOI due to parallel axis theorem with respect to the centroid of the calculated shape
    s     = rp.array(center_of_gravity) - rp.array(centroid) 
    s     = s.ravel()
    I_par = cabin.mass_properties.mass * (rp.array(rp.dot(s, s)) * rp.array(rp.identity(3)) - rp.outer(s, s))             
    
    cabin.mass_properties.moments_of_inertia.tensor =  I_compoment

    return  cabin.mass_properties.moments_of_inertia.tensor, cabin.mass_properties.mass