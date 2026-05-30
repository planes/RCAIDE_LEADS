# RCAIDE/Library/Missions/Common/Update/orientations.py
# 
# 
# Created:  Jul 2023, M. Clarke

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE imports
from RCAIDE.Framework.Core  import Units,  angles_to_dcms, orientation_product, orientation_transpose

# package imports 
import RNUMPY as rp
 
# ----------------------------------------------------------------------------------------------------------------------
#  Update Orientations
# ----------------------------------------------------------------------------------------------------------------------
def orientations(segment):
    
    """ Updates the orientation of the vehicle throughout the mission for each relevant axis
    
        Assumptions:
        This assumes the vehicle has 3 frames: inertial, body, and wind 
        
        Inputs:
        segment.state.conditions:
            frames.inertial.velocity_vector          [meters/second]
            frames.body.inertial_rotations           [Radians]
        segment.analyses.planet.mean_radius [meters]
        state.numerics.time.integrate                [float]
            
        Outputs:
            segment.state.conditions:           
                aerodynamics.angles.alpha         [Radians]
                aerodynamics.angles.beta          [Radians]
                aerodynamics.angles.roll          [Radians]
                frames.body.transform_to_inertial [Radians]
                frames.wind.body_rotations        [Radians]
                frames.wind.transform_to_inertial [Radians]
    

        Properties Used:
        N/A
    """

    # unpack
    conditions              = segment.state.conditions
    V_inertial              = conditions.frames.inertial.velocity_vector
    body_inertial_rotations = conditions.frames.body.inertial_rotations 
    roll_rate               = segment.state.conditions.static_stability.roll_rate      
    pitch_rate              = segment.state.conditions.static_stability.pitch_rate 
    yaw_rate                = segment.state.conditions.static_stability.yaw_rate    
    
    # ------------------------------------------------------------------
    #  Body Frame
    # ------------------------------------------------------------------

    # body frame rotations
    phi  = body_inertial_rotations[:,0,None] 
    beta = conditions.frames.wind.body_rotations[:,2]

    # body frame tranformation matrices
    T_inertial2body = angles_to_dcms(body_inertial_rotations,(2,1,0)) 
    T_body2inertial = orientation_transpose(T_inertial2body) 

    # transform inertial velocity to body frame
    V_body = orientation_product(T_inertial2body,V_inertial)

    # project inertial velocity into body x-z plane
    V_stability = V_body * 1.

    # calculate angle of attack
    alpha = rp.arctan2(V_stability[:,2],V_stability[:,0])[:,None] 

    # pack aerodynamics angles
    conditions.aerodynamics.angles.alpha = alpha
    conditions.aerodynamics.angles.beta  = beta[:,None]
    conditions.aerodynamics.angles.phi   = phi

    # pack transformation tensor
    conditions.frames.body.transform_to_inertial = T_body2inertial 

    # ------------------------------------------------------------------
    #  Wind Frame
    # ------------------------------------------------------------------

    # back calculate wind frame rotations
    # Functional construction to avoid inplace updates
    wind_body_rotations = rp.zeros_like(body_inertial_rotations)
    wind_body_rotations = rp.stack([rp.zeros_like(alpha[:, 0]), alpha[:, 0], beta], axis=1)

    # wind frame tranformation matricies
    T_wind2body     = angles_to_dcms(wind_body_rotations,(2,1,0))       
    T_wind2inertial = orientation_product(T_wind2body,T_body2inertial) 

    # pack wind rotations
    conditions.frames.wind.body_rotations = wind_body_rotations

    # pack transformation tensor
    conditions.frames.wind.transform_to_inertial = T_wind2inertial
    conditions.frames.wind.transform_to_body     = T_wind2body
    
    # ------------------------------------------------------------------
    # Rotation rates 
    # ------------------------------------------------------------------ 
    stability_frame_rotations       =  rp.concatenate((rp.concatenate((roll_rate, pitch_rate), axis=1), yaw_rate), axis=1)
    phi                   = body_inertial_rotations[:, 0]
    theta                 = body_inertial_rotations[:, 1]
    
    # Functional construction of reverse_transformation matrix
    z = rp.zeros_like(phi)
    o = rp.ones_like(phi)
    
    row0 = rp.stack([o, rp.sin(phi)*rp.tan(theta), rp.cos(phi)*rp.tan(theta)], axis=1)
    row1 = rp.stack([z, rp.cos(phi), -rp.sin(phi)], axis=1)
    row2 = rp.stack([z, rp.sin(phi)/rp.cos(theta), rp.cos(phi)/rp.cos(theta)], axis=1)
    
    reverse_transformation = rp.stack([row0, row1, row2], axis=1)
    inertial_rotations              =  orientation_product(reverse_transformation,stability_frame_rotations)
    segment.state.conditions.frames.inertial.angular_velocity_vector = inertial_rotations 
    
    return
         