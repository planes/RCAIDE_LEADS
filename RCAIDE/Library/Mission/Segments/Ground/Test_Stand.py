# RCAIDE/Library/Missions/Segments/Ground/Takeoff.py
# 
# 
# Created:  Jul 2023, M. Clarke  

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE Imports  
import RCAIDE 
import RNUMPY as rp 

# ----------------------------------------------------------------------------------------------------------------------
# unpack unknowns
# ---------------------------------------------------------------------------------------------------------------------- 
def initialize_conditions(segment):
    """Sets the specified conditions which are given for the segment type.

    Assumptions:
    Builds on the initialize conditions for common

    Source:
    N/A

    Inputs:
    segment.throttle                           [unitless]
    conditions.frames.inertial.position_vector [meters]
    conditions.weights.vehicle.mass              [kilogram]

    Outputs:
    conditions.weights.vehicle.mass              [kilogram]
    conditions.frames.inertial.position_vector [unitless]
    conditions.propulsion.throttle             [meters]
    
    Properties Used:
    N/A
    """  

    # use the common initialization # unpack inputs
    alt       = segment.altitude
    time      = segment.time
    v0        = segment.velocity   

    t_initial = segment.state.conditions.frames.inertial.time[0,0]
    t_nondim  = segment.state.numerics.dimensionless.control_points
    time      = rp.max(time)
    charging_time      = t_nondim * ( time ) + t_initial 
    segment.state.conditions.frames.inertial.time = segment.state.conditions.frames.inertial.time.at[:,0].set(charging_time[:,0])
        
    # pack conditions 
    conditions = segment.state.conditions    
    conditions.frames.inertial.velocity_vector = conditions.frames.inertial.velocity_vector.at[:,0].set(v0)
    conditions.freestream.altitude = conditions.freestream.altitude.at[:,0].set(alt)
    conditions.frames.inertial.position_vector = conditions.frames.inertial.position_vector.at[:,2].set(-alt)
    conditions.weights.vehicle.mass = conditions.weights.vehicle.mass.at[:,0].set(segment.analyses.vehicle.mass_properties.takeoff)
    conditions.frames.inertial.position_vector = conditions.frames.inertial.position_vector.at[:,:].set(conditions.frames.inertial.position_vector[0,:][None,:][:,:])
