# RCAIDE/Library/Methods/Geometry/LOPA/LOPA_functions.py
#
#
# Created: Mar 2025, M. Clarke
# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE imports
import RCAIDE
from RCAIDE.Framework.Core import Data
from .LOPA_functions import *
# python functions
import RNUMPY as rp
from copy import  deepcopy
# ----------------------------------------------------------------------------------------------------------------------
#  compute_layout_of_passenger_accommodations
# ----------------------------------------------------------------------------------------------------------------------
def compute_layout_of_passenger_accommodations(fuselage):
    '''
    Creates the layout of passenger accommodations for a vehicle
    '''

    LOPA = rp.empty(( 0, 14))
    offset_x_overall = 0
    
    if len(fuselage.cabins) > 0: 
        # instantiate dimension of LOPA container 
        #  [ x, y, z , length, width, first-cl flag, business-cl flag, economy-cl flag, seat, emergency-row flag, galley/lav flag, type-A exit flag]
        
        side_cabin_offset = 0
        for cabin in fuselage.cabins:
            cabin_LOPA = rp.empty(( 0, 14))
            cabin_number_of_seats = 0
            cabin_class_origin  = [0, 0, 0]
            total_cabin_length = 0
            for cabin_class in cabin.classes:
                seat_data ,cabin_class_origin,cabin_number_of_seats,total_cabin_length  = create_class_seating_map_layout(cabin, cabin_class,cabin_class_origin, side_cabin_offset,cabin_number_of_seats,total_cabin_length)
                side_cabin_offset = cabin.width / 2
                LOPA = rp.vstack((LOPA,seat_data))
                cabin_LOPA = rp.vstack((cabin_LOPA,seat_data))
            cabin.layout_of_passenger_accommodations                     = Data()
            cabin.layout_of_passenger_accommodations.object_coordinates  = cabin_LOPA      
            cabin.layout_of_passenger_accommodations.cabin_x_offset      = offset_x_overall
            cabin.length = total_cabin_length 
            cabin.number_of_seats = cabin_number_of_seats
        
        # determine offset of LOPA from reference point on aircraft (nose)
        for cabin in fuselage.cabins:
            for cabin_class in cabin.classes:
                cabin_class.percentage = cabin_class.length/cabin.length
            if not isinstance(cabin,RCAIDE.Library.Components.Fuselages.Cabins.Side_Cabin):
                offset_x_overall = cabin.origin[0][0]
                 
        fuselage.number_of_seats  = rp.sum(LOPA[:,10])

    fuselage.layout_of_passenger_accommodations                     = Data()
    fuselage.layout_of_passenger_accommodations.object_coordinates  = LOPA        
    fuselage.layout_of_passenger_accommodations.cabin_x_offset      = offset_x_overall

    if LOPA.size > 0 :
        compute_lopa_properties(fuselage, LOPA)    
        
    return 
 
def compute_lopa_properties(fuselage, LOPA):
    # Step 1: plot cabin bounds

    # --- x min ---
    x_vals = LOPA[:, 2]
    y_vals = LOPA[:, 3]
    w_vals = LOPA[:, 5]
    h_vals = LOPA[:, 6]

    x_min_val = rp.min(x_vals)
    x_min_locs = rp.nonzero(x_vals == x_min_val)[0]

    x_min = x_vals[x_min_locs[0]] - w_vals[x_min_locs[0]] / 2
    x_min_y_max = rp.max(y_vals[x_min_locs] + h_vals[x_min_locs] / 2)
    x_min_y_min = rp.min(y_vals[x_min_locs] - h_vals[x_min_locs] / 2)

    x_border_pts = [x_min, x_min]
    y_border_pts = [x_min_y_min, x_min_y_max]

    # --- y max ---
    y_max_val = rp.max(y_vals)
    y_max_locs = rp.nonzero(y_vals == y_max_val)[0]

    y_max = y_vals[y_max_locs[0]] + h_vals[y_max_locs[0]] / 2
    y_max_x_max = rp.max(x_vals[y_max_locs] + w_vals[y_max_locs] / 2)
    y_max_x_min = rp.min(x_vals[y_max_locs] - w_vals[y_max_locs] / 2)

    x_border_pts += [y_max_x_min, y_max_x_max]
    y_border_pts += [y_max, y_max]

    # --- x max ---
    x_max_val = rp.max(x_vals)
    x_max_locs = rp.nonzero(x_vals == x_max_val)[0]

    x_max = x_vals[x_max_locs[0]] + w_vals[x_max_locs[0]] / 2
    x_max_y_max = rp.max(y_vals[x_max_locs] + h_vals[x_max_locs] / 2)
    x_max_y_min = rp.min(y_vals[x_max_locs] - h_vals[x_max_locs] / 2)

    x_border_pts += [x_max, x_max]
    y_border_pts += [x_max_y_max, x_max_y_min]

    # --- y min ---
    y_min_val = rp.min(y_vals)
    y_min_locs = rp.nonzero(y_vals == y_min_val)[0]

    y_min = y_vals[y_min_locs[0]] - h_vals[y_min_locs[0]] / 2
    y_min_x_max = rp.max(x_vals[y_min_locs] + w_vals[y_min_locs] / 2)
    y_min_x_min = rp.min(x_vals[y_min_locs] - w_vals[y_min_locs] / 2)

    x_border_pts += [y_min_x_max, y_min_x_min]
    y_border_pts += [y_min, y_min]

    # Convert to arrays
    x_border_pts = rp.array(x_border_pts)
    y_border_pts = rp.array(y_border_pts)

    # Remove negative y (port side)
    port_idxs = rp.nonzero(y_border_pts < 0)[0]
    starboard_x = rp.delete(x_border_pts, port_idxs)
    starboard_y = rp.delete(y_border_pts, port_idxs)

    # Save results
    fuselage.layout_of_passenger_accommodations.cabin_area_coordinates = \
        rp.vstack((starboard_x[None, :], starboard_y[None, :])).T

    fuselage.layout_of_passenger_accommodations.cabin_length = rp.max(starboard_x)
    fuselage.layout_of_passenger_accommodations.cabin_width = 2 * rp.max(starboard_y)


# def compute_lopa_properties(fuselage, LOPA): 
#     # Step 1: plot cabin bounds
#     # get points at x min
#     x_min_locs   =  rp.where( LOPA[:,2] == min(LOPA[:,2]))[0]
#     x_min        =  LOPA[x_min_locs[0],2] -  LOPA[x_min_locs[0],5]/2
#     x_min_y_max  =  max(LOPA[x_min_locs,3] + LOPA[x_min_locs,6]/2 )
#     x_min_y_min  =  min(LOPA[x_min_locs,3] - LOPA[x_min_locs,6]/2 )
#     x_border_pts = [x_min, x_min]
#     y_border_pts = [x_min_y_min, x_min_y_max]
    
#     # get points at y max
#     y_max_locs   =  rp.where( LOPA[:,3] == max(LOPA[:,3]))[0]
#     y_max        =  LOPA[y_max_locs[0],3] + LOPA[y_max_locs[0],6]/2
#     y_max_x_max  =  max(LOPA[y_max_locs,2] + LOPA[y_max_locs[0],5]/2)
#     y_max_x_min  =  min(LOPA[y_max_locs,2] - LOPA[y_max_locs[0],5]/2)
#     x_border_pts.append(y_max_x_min)
#     x_border_pts.append(y_max_x_max)
#     y_border_pts.append(y_max)
#     y_border_pts.append(y_max)
    
#     # get points at x max
#     x_max_locs   =  rp.where( LOPA[:,2] == max(LOPA[:,2]))[0]
#     x_max        =  LOPA[x_max_locs[0],2] + LOPA[x_max_locs[0],5]/2
#     x_max_y_max  =  max(LOPA[x_max_locs,3] + LOPA[x_max_locs,6]/2)
#     x_max_y_min  =  min(LOPA[x_max_locs,3] - LOPA[x_max_locs,6]/2)
#     x_border_pts.append(x_max)
#     x_border_pts.append(x_max)
#     y_border_pts.append(x_max_y_max)
#     y_border_pts.append(x_max_y_min)
    
#     # get points at y min
#     y_min_locs   =  rp.where( LOPA[:,3] == min(LOPA[:,3]))[0]
#     y_min        =  LOPA[y_min_locs[0],3] - LOPA[y_min_locs[0],6]/2
#     y_min_x_max  =  max(LOPA[y_min_locs,2] + LOPA[y_min_locs[0],5]/2)
#     y_min_x_min  =  min(LOPA[y_min_locs,2] - LOPA[y_min_locs[0],5]/2)
#     x_border_pts.append(y_min_x_max)
#     x_border_pts.append(y_min_x_min)
#     y_border_pts.append(y_min)
#     y_border_pts.append(y_min)
    
#     # loop through points and determine if there are duplicates
#     y_border_pts = rp.array(y_border_pts)
#     x_border_pts = rp.array(x_border_pts)
    
#     # cut where y is negative
#     port_idxs  =  rp.where(y_border_pts<0)[0]
#     starboard_x_points = rp.delete(x_border_pts, port_idxs)
#     starboard_y_points = rp.delete(y_border_pts, port_idxs)
    
#     fuselage.layout_of_passenger_accommodations.cabin_area_coordinates = rp.vstack((starboard_x_points[None,:],starboard_y_points[None, :])).T
#     fuselage.layout_of_passenger_accommodations.cabin_length           = max(starboard_x_points)
#     fuselage.layout_of_passenger_accommodations.cabin_width            = 2*max(starboard_y_points)
    
#     return

def create_class_seating_map_layout(cabin,cabin_class,cabin_class_origin, side_cabin_offset,cabin_number_of_seats,cabin_length):
    s_y_coord, cabin_class_origin = get_seat_y_coords(cabin, cabin_class,cabin_class_origin)
    s_x_coord,object_type, cabin_class_origin,cabin_length = get_seat_x_coords(cabin, cabin_class,cabin_class_origin,cabin_length)
    # concatenate arrays
    length   =  cabin_class.seat_length * rp.ones_like(s_x_coord)
    length = length.at[object_type[:,2] == 1].set(cabin.galley_lavatory_length)
    length = length.at[object_type[:,3] == 1].set(cabin.type_A_door_length)
    X_coords   = rp.atleast_2d((rp.tile(s_x_coord[:,None], (1, len(s_y_coord)))).flatten()).T
    Y_coords   = rp.atleast_2d((rp.tile(s_y_coord[None,:], (len(s_x_coord), 1))).flatten()).T
    Z_coords   = rp.atleast_2d((rp.zeros_like(Y_coords)).flatten()).T
    length     = rp.atleast_2d((rp.tile(length[:,None], (1, len(s_y_coord)))).flatten()).T
    width      = cabin_class.seat_width * rp.ones_like(Z_coords)
    n_rows     = cabin_class.number_of_rows * rp.ones_like(Z_coords)
    n_seats_y  = cabin_class.number_of_seats_abrest  * rp.ones_like(Z_coords)
    object_vec = rp.repeat(object_type, len(s_y_coord), axis=0)
    # cabin class flags
    F_c  =  rp.zeros_like(Z_coords)
    B_c  =  rp.zeros_like(Z_coords)
    E_c  =  rp.zeros_like(Z_coords)
    if type(cabin_class) == RCAIDE.Library.Components.Fuselages.Cabins.Classes.First:
        F_c  =  rp.ones_like(Z_coords)
    elif type(cabin_class) == RCAIDE.Library.Components.Fuselages.Cabins.Classes.Business:
        B_c  =  rp.ones_like(Z_coords)
    elif type(cabin_class) == RCAIDE.Library.Components.Fuselages.Cabins.Classes.Economy:
        E_c  =  rp.ones_like(Z_coords)
    # [n_rows , n_seats_y, x, y, z , length, width, first-cl flag, business-cl flag, economy-cl flag, [seat, emergency-row flag, galley/lav flag, type-A exit flag]]
    seat_data = rp.hstack((n_rows , n_seats_y, X_coords,Y_coords, Z_coords, length ,width,F_c,B_c,E_c,object_vec))
    if type(cabin) == RCAIDE.Library.Components.Fuselages.Cabins.Side_Cabin:
        seat_data = seat_data.at[:, 3].add(cabin.width / 2)
        seat_data  = update_seat_map_layout_using_cabin_taper(seat_data,cabin)
        seat_data = seat_data.at[:, 3].add(side_cabin_offset  +  cabin_class.y_offset_distance)
        # make copy about center
        seat_data_        = deepcopy(seat_data)
        seat_data_ = seat_data_.at[:, 3].multiply(-1)
        seat_data         = rp.vstack((seat_data,seat_data_))
    cabin_class.number_of_seats = int(rp.sum(seat_data[:,10]))
    cabin_number_of_seats       += int(cabin_class.number_of_seats)
    return seat_data ,cabin_class_origin,cabin_number_of_seats,cabin_length