# RCAIDE/Library/Methods/Mass_Properties/Moment_of_Inertia/compute_bwb_center_of_gravity.py 
# 
# Created:  January 2026, S. Shekar, A. Molloy M. Clarke,  
 
# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
import RCAIDE

# package imports 
import RNUMPY as rp
from RCAIDE.Library.Methods.Geometry.Mesh import Mesh, get_convex_hull, clip_polygon_x, get_polygon_area, get_polygon_centroid

# ----------------------------------------------------------------------------------------------------------------------
#  Compute Blended Wing Body Center of Gravity
# ----------------------------------------------------------------------------------------------------------------------  
def compute_bwb_center_of_gravity(bwb_wing, vehicle): 
    ''' computes the moment of inertia tensor for a blended wing body about a given center of gravity.
    Includes the ability to model a  wing fuel tank as a condensed wing 
    
    Inputs:
    - Wing 
    - Vehicle 

    Outputs:
    - wing center of gravity 

    Properties Used:
    N/A
    '''

    center_body_segs = []
    wing_segs        = []
    for segment in bwb_wing.segments: 
        if isinstance(segment, RCAIDE.Library.Components.Wings.Segments.Blended_Wing_Body_Fuselage_Segment): 
            center_body_segs.append(segment.tag)
        else:
            if segment.percent_span_location != 1.0:
                wing_segs.append(segment.tag)
    wing_segs.insert(0, center_body_segs[-1])
            
    # compute cabin moment of inertia 
    compute_center_body_center_of_gravity(bwb_wing,center_body_segs)
    
    # compute aft cabin moment of inertia 
    compute_aft_center_body_center_of_gravity(bwb_wing,center_body_segs)

    # compute wing moment of inertia 
    compute_bwb_wing_center_of_gravity(bwb_wing,wing_segs) 

    return bwb_wing.mass_properties.center_of_gravity 

def compute_bwb_wing_center_of_gravity(bwb_wing,seg_keys):

    mass = bwb_wing.mass_properties.mass

    #populate the wing segment properties and other things 
    segment_meshes = [] 
    for i in range(len(seg_keys)-1):
        # compute volume and assume unit density to get mass
        inner_segment = bwb_wing.segments[seg_keys[i]]
        outer_segment = bwb_wing.segments[seg_keys[i+1]]


        x_in = rp.array(inner_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][0]
        y_in = rp.array(inner_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][2]
        x_out = rp.array(outer_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][0]
        y_out = rp.array(outer_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][2]


        # STEP 1: Build 3D point clouds for both sections
        x1, y1 = x_in, y_in
        x2, y2 = x_out, y_out

        pts1 = rp.column_stack((x1[:-1], y1[:-1], rp.zeros(len(x1)-1)))   # z = 0
        pts2 = rp.column_stack((x2[:-1], y2[:-1], rp.ones(len(x2)-1) * L)) # z = L

        # STEP 2: Combine all points
        all_pts = rp.vstack([pts1, pts2])

        # STEP 3: Convex hull → watertight volume mesh
        solid_segment = get_convex_hull(all_pts)

        # Apply spanwise translation AFTER orientation fix
        T = rp.eye(4)
        T = T.at[0, 3].set(0.0)
        T = T.at[1, 3].set(0.0)
        T = T.at[2, 3].set(inner_segment.percent_span_location * bwb_wing.spans.projected/2)
        solid_segment.apply_transform(T)
        # Rotate 90 degrees around X axis to match RCAIDE convention
        R = rp.eye(4)
        R = R.at[1, 1].set(0.0)
        R = R.at[1, 2].set(-1.0)
        R = R.at[2, 1].set(1.0)
        R = R.at[2, 2].set(0.0)
        solid_segment.apply_transform(R)

        segment_meshes.append(solid_segment)
    
    combinde_mesh = Mesh.concatenate(segment_meshes)
    # Reflect across the YZ plane (mirror X)
    Ry = rp.diag(rp.array([1.0, -1.0, 1.0]))   # reflection matrix

    # 1. copy the mesh
    combined_mesh_sym = deepcopy(combinde_mesh)

    # 2. apply the mirror transform
    Ry_cast = rp.array(Ry, dtype=combined_mesh_sym.vertices.dtype)
    combined_mesh_sym.vertices = (Ry_cast @ combined_mesh_sym.vertices.T).T

    # 3. fix face orientation (reverse winding)
    combined_mesh_sym.faces = combined_mesh_sym.faces[:, ::-1]

    # 4. concatenate original + mirrored
    combined_mesh_full = Mesh.concatenate([combinde_mesh, combined_mesh_sym])
    combined_mesh_full.density = mass / combined_mesh_full.volume
    I        = combined_mesh_full.moment_inertia
    centroid = rp.array(combined_mesh_full.centroid)
    centroid[1] = 0 
      
    # store values 
    bwb_wing.mass_properties.center_of_gravity         =  [centroid.tolist()]
    bwb_wing.mass_properties.moments_of_inertia.tensor =  I
    return   

def compute_aft_center_body_center_of_gravity(bwb_wing,seg_keys):
    mass          = bwb_wing.aft_center_body.mass_properties.mass
    # origin_x      = bwb_wing.layout_of_passenger_accommodations.object_coordinates[-1][2] + bwb_wing.layout_of_passenger_accommodations.cabin_x_offset
    cabin_length  = bwb_wing.layout_of_passenger_accommodations.object_coordinates[-1][2] + bwb_wing.layout_of_passenger_accommodations.cabin_x_offset 

    segment_meshes = [] 
    for i in range(len(seg_keys)-1):
        # compute volume and assume unit density to get mass
        inner_segment = bwb_wing.segments[seg_keys[i]]
        outer_segment = bwb_wing.segments[seg_keys[i+1]]


        x_in = rp.array(inner_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][0]
        y_in = rp.array(inner_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][2]
        x_out = rp.array(outer_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][0]
        y_out = rp.array(outer_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][2]

        x_out, y_out = clip_polygon_x(x_out, y_out, x_min=cabin_length)
        x_in, y_in   = clip_polygon_x(x_in, y_in, x_min=cabin_length)

        # Compute segment span length
        L = (outer_segment.percent_span_location - inner_segment.percent_span_location) * bwb_wing.spans.projected/2
        # STEP 1: Build 3D point clouds for both sections
        x1, y1 = x_in, y_in
        x2, y2 = x_out, y_out

        pts1 = rp.column_stack((x1[:-1], y1[:-1], rp.zeros(len(x1)-1)))   # z = 0
        pts2 = rp.column_stack((x2[:-1], y2[:-1], rp.ones(len(x2)-1) * L)) # z = L

        # STEP 2: Combine all points
        all_pts = rp.vstack([pts1, pts2])

        # STEP 3: Convex hull → watertight volume mesh
        solid_segment = get_convex_hull(all_pts)

        # Apply spanwise translation AFTER orientation fix
        T = rp.eye(4)
        T = T.at[0, 3].set(0.0)
        T = T.at[1, 3].set(0.0)
        T = T.at[2, 3].set(inner_segment.percent_span_location * bwb_wing.spans.projected/2)
        solid_segment.apply_transform(T)
        # Rotate 90 degrees around X axis to match RCAIDE convention
        R = rp.eye(4)
        R = R.at[1, 1].set(0.0)
        R = R.at[1, 2].set(-1.0)
        R = R.at[2, 1].set(1.0)
        R = R.at[2, 2].set(0.0)
        solid_segment.apply_transform(R)

        segment_meshes.append(solid_segment)
    
    combinde_mesh = Mesh.concatenate(segment_meshes)
    # Reflect across the YZ plane (mirror X)
    Ry = rp.diag(rp.array([1.0, -1.0, 1.0]))   # reflection matrix

    # 1. copy the mesh
    combined_mesh_sym = deepcopy(combinde_mesh)

    # 2. apply the mirror transform
    Ry_cast = rp.array(Ry, dtype=combined_mesh_sym.vertices.dtype)
    combined_mesh_sym.vertices = (Ry_cast @ combined_mesh_sym.vertices.T).T

    # 3. fix face orientation (reverse winding)
    combined_mesh_sym.faces = combined_mesh_sym.faces[:, ::-1]

    # 4. concatenate original + mirrored
    combined_mesh_full         = Mesh.concatenate([combinde_mesh, combined_mesh_sym])
    combined_mesh_full.density = mass / combined_mesh_full.volume
    I                          = combined_mesh_full.moment_inertia
    centroid                   = rp.array(combined_mesh_full.centroid)
    centroid[1] = 0

    # store values 
    bwb_wing.aft_center_body.mass_properties.center_of_gravity         =  [centroid.tolist()]
    bwb_wing.aft_center_body.mass_properties.moments_of_inertia.tensor =  I
    
    return  

def compute_center_body_center_of_gravity(bwb_wing,seg_keys): 
    mass          = bwb_wing.center_body.mass_properties.mass
    origin_x      = bwb_wing.layout_of_passenger_accommodations.cabin_x_offset
    cabin_length  = bwb_wing.layout_of_passenger_accommodations.object_coordinates[-1][2] + origin_x

    segment_meshes = [] 
    for i in range(len(seg_keys)-1):
        # compute volume and assume unit density to get mass
        inner_segment = bwb_wing.segments[seg_keys[i]]
        outer_segment = bwb_wing.segments[seg_keys[i+1]]


        x_in = rp.array(inner_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][0]
        y_in = rp.array(inner_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *inner_segment.root_chord_percent + inner_segment.origin[0][2]
        x_out = rp.array(outer_segment.airfoil.geometry.x_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][0]
        y_out = rp.array(outer_segment.airfoil.geometry.y_coordinates)[:-1] * bwb_wing.chords.root *outer_segment.root_chord_percent+ outer_segment.origin[0][2]

        x_out, y_out = clip_polygon_x(x_out, y_out, x_max=cabin_length)
        x_in, y_in   = clip_polygon_x(x_in, y_in, x_max=cabin_length)
        
        A_1 = get_polygon_area(x_in, y_in)
        A_2 = get_polygon_area(x_out, y_out)

        # Compute segment span length
        L = (outer_segment.percent_span_location - inner_segment.percent_span_location) * bwb_wing.spans.projected/2
        # STEP 1: Build 3D point clouds for both sections
        x1, y1 = x_in, y_in
        x2, y2 = x_out, y_out

        pts1 = rp.column_stack((x1[:-1], y1[:-1], rp.zeros(len(x1)-1)))   # z = 0
        pts2 = rp.column_stack((x2[:-1], y2[:-1], rp.ones(len(x2)-1) * L)) # z = L

        # STEP 2: Combine all points
        all_pts = rp.vstack([pts1, pts2])

        # STEP 3: Convex hull → watertight volume mesh
        solid_segment = get_convex_hull(all_pts)

        # Apply spanwise translation AFTER orientation fix
        T = rp.eye(4)
        T = T.at[0, 3].set(0.0)
        T = T.at[1, 3].set(0.0)
        T = T.at[2, 3].set(inner_segment.percent_span_location * bwb_wing.spans.projected/2)
        solid_segment.apply_transform(T)
        # Rotate 90 degrees around X axis to match RCAIDE convention
        R = rp.eye(4)
        R = R.at[1, 1].set(0.0)
        R = R.at[1, 2].set(-1.0)
        R = R.at[2, 1].set(1.0)
        R = R.at[2, 2].set(0.0)
        solid_segment.apply_transform(R)

        segment_meshes.append(solid_segment)
    
    combinde_mesh = Mesh.concatenate(segment_meshes)
    # Reflect across the YZ plane (mirror X)
    Ry = rp.diag(rp.array([1.0, -1.0, 1.0]))   # reflection matrix

    # 1. copy the mesh
    combined_mesh_sym = deepcopy(combinde_mesh)

    # 2. apply the mirror transform
    Ry_cast = rp.array(Ry, dtype=combined_mesh_sym.vertices.dtype)
    combined_mesh_sym.vertices = (Ry_cast @ combined_mesh_sym.vertices.T).T

    # 3. fix face orientation (reverse winding)
    combined_mesh_sym.faces = combined_mesh_sym.faces[:, ::-1]

    # 4. concatenate original + mirrored
    combined_mesh_full         = Mesh.concatenate([combinde_mesh, combined_mesh_sym])
    combined_mesh_full.density = mass / combined_mesh_full.volume
    I                          = combined_mesh_full.moment_inertia
    centroid                   = rp.array(combined_mesh_full.centroid)
    centroid[1] = 0
    # store values 
    bwb_wing.center_body.mass_properties.center_of_gravity         =  [centroid.tolist()]
    bwb_wing.center_body.mass_properties.moments_of_inertia.tensor =  I
    
    return   