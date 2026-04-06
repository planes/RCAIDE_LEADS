# RCAIDE/Library/Methods/Geometry/Planform/compute_segment_meshes.py 
# 
# Created:  Dec 2025, M. Clarke 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ---------------------------------------------------------------------------------------------------------------------- 

# package imports 
import RNUMPY as rp
from RCAIDE.Library.Methods.Geometry.Mesh import get_convex_hull


# ----------------------------------------------------------------------------------------------------------------------
#  Compute segment meshes 
# ---------------------------------------------------------------------------------------------------------------------- 
def compute_segment_meshes(x_in,y_in, x_out, y_out, L, spanwise_shift):

    # STEP 1: Build 3D point clouds for both sections
    pts1 = rp.column_stack((x_in, y_in, rp.zeros(len(x_in))))   # z = 0
    pts2 = rp.column_stack((x_out, y_out, rp.ones(len(x_out)) * L)) # z = L

    # STEP 2: Combine all points
    all_pts = rp.vstack([pts1, pts2])

    # STEP 3: Convex hull → watertight volume mesh
    solid_segment = get_convex_hull(all_pts)

    # Apply spanwise translation AFTER orientation fix
    T = rp.eye(4)
    T[0, 3] = 0.0
    T[1, 3] = 0.0
    T[2, 3] = spanwise_shift
    solid_segment.apply_transform(T)
    
    # Rotate 90 degrees around X axis to match RCAIDE convention
    R = rp.eye(4)
    cos_a = 0.0 # cos(90)
    sin_a = 1.0 # sin(90)
    R[1, 1] = cos_a
    R[1, 2] = -sin_a
    R[2, 1] = sin_a
    R[2, 2] = cos_a
    solid_segment.apply_transform(R)
    
    return solid_segment