# RCAIDE/Library/Missions/Segments/Single_Point/Set_Speed_Set_Altitude_No_Propulsion.py
# 
# 
# Created:  Jul 2023, M. Clarke 
 
# ----------------------------------------------------------------------------------------------------------------------  
#  Initialize Conditions
# ----------------------------------------------------------------------------------------------------------------------  

import RNUMPY as rp

# ----------------------------------------------------------------------------------------------------------------------
#  Set_Speed_Set_Altitude_No_Propulsion
# ---------------------------------------------------------------------------------------------------------------------- 
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    A fixed speed and altitude

    Source:
    N/A

    Inputs:
    segment.altitude                            [meters]
    segment.air_speed                           [meters/second]
    segment.z_accel                             [meters/second^2]

    Outputs:
    conditions.frames.inertial.acceleration_vector [meters/second^2]
    conditions.frames.inertial.velocity_vector     [meters/second]
    conditions.frames.inertial.position_vector     [meters]
    conditions.freestream.altitude                 [meters]
    conditions.frames.inertial.time                [seconds]

    Properties Used:
    N/A
    """      
    

    # unpack
    alt                   = segment.altitude
    air_speed             = segment.air_speed  
    sideslip              = segment.sideslip_angle
    linear_acceleration_x = segment.linear_acceleration_x
    linear_acceleration_y = segment.linear_acceleration_y
    linear_acceleration_z = segment.linear_acceleration_z 
    roll_rate             = segment.roll_rate
    pitch_rate            = segment.pitch_rate
    yaw_rate              = segment.yaw_rate
    
    # check for initial altitude
    if alt is None:
        if not segment.state.initials: raise AttributeError('altitude not set')
        alt = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
    
    # pack
    air_speed_x                                                   = rp.cos(sideslip)*air_speed 
    air_speed_y                                                   = rp.sin(sideslip)*air_speed 
    segment.state.conditions.freestream.altitude = segment.state.conditions.freestream.altitude.at[:,0].set(alt)
    segment.state.conditions.frames.inertial.position_vector = segment.state.conditions.frames.inertial.position_vector.at[:,2].set(-alt) # z points down
    segment.state.conditions.frames.inertial.velocity_vector = segment.state.conditions.frames.inertial.velocity_vector.at[:,0].set(air_speed_x)
    segment.state.conditions.frames.inertial.velocity_vector = segment.state.conditions.frames.inertial.velocity_vector.at[:,1].set(air_speed_y)
    segment.state.conditions.frames.inertial.acceleration_vector  = rp.array([[linear_acceleration_x,linear_acceleration_y,linear_acceleration_z]])  
    segment.state.conditions.static_stability.roll_rate = segment.state.conditions.static_stability.roll_rate.at[:,0].set(roll_rate)
    segment.state.conditions.static_stability.pitch_rate = segment.state.conditions.static_stability.pitch_rate.at[:,0].set(pitch_rate)
    segment.state.conditions.static_stability.yaw_rate = segment.state.conditions.static_stability.yaw_rate.at[:,0].set(yaw_rate)
    
    return 