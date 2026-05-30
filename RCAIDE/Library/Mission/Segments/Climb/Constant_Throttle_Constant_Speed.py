# RCAIDE/Library/Missions/Segments/Climb/Constant_Throttle_Constant_Speed.py
# 
# 
# Created:  Jul 2023, M. Clarke

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ---------------------------------------------------------------------------------------------------------------------- 

# Package imports  
import RNUMPY as rp
 
# ----------------------------------------------------------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------------------------------------------------------
def unpack_body_angle(segment):
    """
    Unpacks and sets the proper value for body angle

    Parameters
    ----------
    segment : Segment
        The mission segment being analyzed

    Notes
    -----
    This function handles the initialization of the body angle for a climb segment
    with constant throttle and constant speed.

    **Required Segment Components**

    segment:
        state:
            unknowns:
                body_angle : array
                    Aircraft body angle [rad]
            conditions:
                frames:
                    body:
                        transform_to_inertial : array
                            Rotation matrix from body to inertial frame

    Returns
    -------
    None
        Updates segment conditions directly:
        - conditions.frames.body.transform_to_inertial
    """          
    
    ctrls    = segment.assigned_control_variables 

    # Body Angle Control    
    if ctrls.body_angle.active: 
        segment.state.conditions.frames.body.inertial_rotations = segment.state.conditions.frames.body.inertial_rotations.at[:,1].set(segment.state.unknowns.body_angle[:,0])
    else:
        segment.state.conditions.frames.body.inertial_rotations = segment.state.conditions.frames.body.inertial_rotations.at[:,1].set(segment.angle_of_attack)
        
    # Bank Angle 
    if ctrls.bank_angle.active: 
        segment.state.conditions.frames.body.inertial_rotations = segment.state.conditions.frames.body.inertial_rotations.at[:,0].set(segment.state.unknowns.bank_angle[:,0])
    else:
        segment.state.conditions.frames.body.inertial_rotations = segment.state.conditions.frames.body.inertial_rotations.at[:,0].set(segment.bank_angle)
    
    # Sideslip Angle  
    segment.state.conditions.frames.wind.body_rotations = segment.state.conditions.frames.wind.body_rotations.at[:,2].set(segment.sideslip_angle)
       
    # Heading Angle  
    segment.state.conditions.frames.body.inertial_rotations = segment.state.conditions.frames.body.inertial_rotations.at[:,2].set(segment.state.conditions.frames.planet.true_heading[:,0])
         
# ----------------------------------------------------------------------
#  Initialize Conditions
# ----------------------------------------------------------------------

def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.
    
    Assumptions:
    Constant throttle estting, with a constant true airspeed

    Source:
    N/A

    Inputs:
    segment.air_speed                                   [meters/second] 
    segment.altitude_start                              [meters]
    segment.altitude_end                                [meters]
    segment.state.numerics.dimensionless.control_points [Unitless]
    conditions.freestream.density                       [kilograms/meter^3]

    Outputs:
    conditions.frames.inertial.velocity_vector          [meters/second]

    Properties Used:
    N/A
    """         
    
    # unpack 
    alt0       = segment.altitude_start 
    v_mag      = segment.air_speed 
    beta       = segment.sideslip_angle
    alpha      = segment.state.unknowns.wind_angle[:,0][:,None]
    theta      = segment.state.unknowns.body_angle[:,0][:,None]
    conditions = segment.state.conditions  

    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]
   
    # Flight path angle
    gamma = theta-alpha

    # process
    v_x =   rp.cos(beta) *v_mag * rp.cos(gamma)
    v_y =   rp.sin(beta) *v_mag * rp.cos(gamma)
    v_z = -v_mag * rp.sin(gamma) # z points down

    # pack
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,0].set(v_x[:,0])
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,1].set(v_y[:,0])
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,2].set(v_z[:,0])
 

def update_differentials_altitude(segment):
    """On each iteration creates the differentials and integration funcitons from knowns about the problem. Sets the time at each point. Must return in dimensional time, with t[0] = 0
    
    Assumptions:
    Constant throttle setting, with a constant true airspeed.

    Source:
    N/A

    Inputs:
    segment.climb_angle                         [radians]
    state.conditions.frames.inertial.velocity_vector [meter/second]
    segment.altitude_start                      [meters]
    segment.altitude_end                        [meters]

    Outputs:
    state.conditions.frames.inertial.time       [seconds]
    conditions.frames.inertial.position_vector  [meters]
    conditions.freestream.altitude              [meters]

    Properties Used:
    N/A
    """   

    # unpack
    t = segment.state.numerics.dimensionless.control_points
    I = segment.state.numerics.dimensionless.integrate
    
    # Unpack segment initials
    alt0       = segment.altitude_start 
    altf       = segment.altitude_end    
    conditions = segment.state.conditions  
    v          = segment.state.conditions.frames.inertial.velocity_vector
    
    # check for initial altitude
    if alt0 is None:
        if not segment.state.initials: raise AttributeError('initial altitude not set')
        alt0 = -1.0 *segment.state.initials.conditions.frames.inertial.position_vector[-1,2]    
    
    # get overall time step
    vz = -v[:,2,None] # Inertial velocity is z down
    dz = altf- alt0    
    dt = dz / rp.dot(I[-1,:],vz)[-1] # maintain column array
    
    # Integrate vz to get altitudes
    alt = alt0 + rp.dot(I*dt,vz)

    # rescale operators
    t = t * dt

    # pack
    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    segment.state.conditions.frames.inertial.time = segment.state.conditions.frames.inertial.time.at[:,0].set(t_initial + t[:,0])
    conditions.frames.inertial.position_vector = conditions.frames.inertial.position_vector.at[:,2].set(-alt[:,0]) # z points down
    conditions.freestream.altitude = conditions.freestream.altitude.at[:,0].set(alt[:,0]) # positive altitude in this context    

    return

# ----------------------------------------------------------------------
#  Update Velocity Vector from Wind Angle
# ----------------------------------------------------------------------

def update_velocity_vector_from_wind_angle(segment):
    
    # unpack
    conditions = segment.state.conditions 
    v_mag      = segment.air_speed 
    beta       = segment.sideslip_angle
    alpha      = segment.state.unknowns.wind_angle[:,0][:,None]
    theta      = segment.state.unknowns.body_angle[:,0][:,None]
    
    # Flight path angle
    gamma = theta-alpha

    # process
    v_x =   rp.cos(beta) *v_mag * rp.cos(gamma)
    v_y =   rp.sin(beta) *v_mag * rp.cos(gamma)
    v_z = -v_mag * rp.sin(gamma) # z points down

    # pack
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,0].set(v_x[:,0])
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,1].set(v_y[:,0])
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,2].set(v_z[:,0])

    return conditions
