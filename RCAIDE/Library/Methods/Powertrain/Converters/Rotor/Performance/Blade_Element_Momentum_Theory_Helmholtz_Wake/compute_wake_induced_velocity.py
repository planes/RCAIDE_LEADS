# RCAIDE/Library/Methods/Powertrain/Converters/Rotor/Performance/BEMT_Hemholtz_Vortex_Theory/compute_wake_induced_velocity.py
# 
# Created:   Jun 2021, R. Erhard 
# Modified:  Jun 2024, M. Clarke

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE imports  
import RNUMPY as rp 
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------------------------------------------------- 
#  compute_induced_velocity
# ----------------------------------------------------------------------------------------------------------------------   
def compute_wake_induced_velocity(rotor, rotor_conditions, evaluation_points, ctrl_pts, identical_flag=False):
    """
    Computes the velocity induced by a rotor wake on specified evaluation points.
    
    Parameters
    ----------
    rotor : RCAIDE.Library.Components.Powertrain.Converters.Rotor
        Rotor component with the following attributes:
            - tip_radius : float
                Tip radius of the rotor [m]
            - origin : array_like
                Origin coordinates of the rotor [m, m, m]
            - rotation : int
                Rotation direction (1 for CCW, -1 for CW)
    rotor_conditions : Data
        Rotor operating conditions with:
            - disc_radial_distribution : array_like
                Radial distribution on the disc [m]
            - blade_axial_induced_velocity : array_like
                Axial induced velocity at the blade [m/s]
            - blade_tangential_induced_velocity : array_like
                Tangential induced velocity at the blade [m/s]
    evaluation_points : Data
        Points where induced velocities are to be evaluated:
            - XC : array_like
                X-coordinates of evaluation points (vehicle frame) [m]
            - YC : array_like
                Y-coordinates of evaluation points (vehicle frame) [m]
            - ZC : array_like
                Z-coordinates of evaluation points (vehicle frame) [m]
    ctrl_pts : int
        Number of control points in segment
    identical_flag : bool, optional
        Flag indicating if evaluation points are identical to rotor points, default False
    
    Returns
    -------
    rotor_V_wake_ind : array_like
        Induced velocities at evaluation points, shape (ctrl_pts, n_cp, 3)
        where n_cp is the number of evaluation points
    
    Notes
    -----
    This function calculates the velocity induced by a rotor wake at specified evaluation
    points using a simplified wake contraction model. It is particularly useful for
    analyzing rotor-rotor interactions and rotor-airframe interactions.
    
    The computation follows these steps:
        1. Extract rotor parameters and induced velocities at the blade
        2. Identify evaluation points within the rotor's influence range
        3. Calculate the distance of evaluation points from the rotor plane
        4. Apply wake contraction model based on McCormick's formulation
        5. Interpolate axial and tangential induced velocities at evaluation points
        6. Apply contraction factor to scale induced velocities
        7. Adjust sign of tangential velocities based on position relative to hub
    
    **Major Assumptions**
        * The wake contracts following McCormick's formulation
        * Induced velocities are only calculated for points within the rotor's radial range
        * Inboard and outboard regions of the rotor are treated separately
        * Tangential induced velocities change sign across the hub center
    
    **Theory**
    The wake contraction factor (kd) is calculated as:
    
    .. math::
        k_d = 1 + \\frac{s}{\\sqrt{s^2 + R^2}}
    
    where:
        - s is the distance from the rotor plane
        - R is the rotor tip radius
    
    The induced velocities at evaluation points are then scaled by this contraction factor:
    
    .. math::
        v_{a,new} = k_d \\cdot v_a(y)
    
    .. math::
        v_{t,new} = k_d \\cdot v_t(y)
    
    where:
        - v_a is the axial induced velocity
        - v_t is the tangential induced velocity
        - y is the radial position
    
    References
    ----------
    [1] McCormick, B.W., "Aerodynamics of V/STOL Flight", Academic Press, 1969
    
    See Also
    --------
    RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake.wake_model
    """

    # extract vortex distribution
    n_cp = len(evaluation_points.XC[0])

    # initialize rotor wake induced velocities
    rotor_V_wake_ind = rp.zeros((ctrl_pts,n_cp,3))

    R            = rotor.tip_radius
    r            = rotor_conditions.disc_radial_distribution[0,:,0]

    # Ignore points within hub or outside tip radius
    hub_y_center = rotor.origin[0][1]
    inboard_r    = rp.flip(hub_y_center - r) 
    outboard_r   = hub_y_center + r 
    rotor_y_range = rp.append(inboard_r, outboard_r)

    # Identify points inside the inboard and outboard wake regions
    bool_inboard  = (evaluation_points.YC > inboard_r[0])  & (evaluation_points.YC < inboard_r[-1])
    bool_outboard = (evaluation_points.YC > outboard_r[0]) & (evaluation_points.YC < outboard_r[-1])
    bool_in_range = bool_inboard | bool_outboard

    # 1. Compute 's' and 'kd' over the ENTIRE array (no reshaping or slicing needed)
    s_full  = evaluation_points.XC - rotor.origin[0][0]
    kd_full = 1.0 + s_full / rp.sqrt(s_full**2 + R**2)

    # 2. Extract velocities and setup interpolators (Unchanged)
    va = rotor_conditions.blade_axial_induced_velocity[0]
    vt = rotor_conditions.blade_tangential_induced_velocity[0]

    rotation = 1 if rotor.clockwise_rotation else -1

    va_y_range  = rp.append(rp.flipud(va), va)
    vt_y_range  = rp.append(rp.flipud(vt), vt) * rotation
    va_interp   = interp1d(rotor_y_range, va_y_range)
    vt_interp   = interp1d(rotor_y_range, vt_y_range)

    # 3. Clamp YC coordinates so the interpolator doesn't throw out-of-bounds errors 
    # for points outside the wake (we will zero them out in step 5 anyway).
    y_safe = rp.clip(evaluation_points.YC, rotor_y_range[0], rotor_y_range[-1])

    # 4. Compute induced velocities for the ENTIRE array
    va_new_full = kd_full * va_interp(y_safe)
    
    # Use rp.where to conditionally flip the tangential velocity out-of-place!
    inboard_bools  = (evaluation_points.YC < hub_y_center)
    vt_interp_full = vt_interp(y_safe)
    vt_new_full    = kd_full * rp.where(inboard_bools, -vt_interp_full, vt_interp_full)

    # 5. Apply the bool_in_range mask to zero out the points we ignored
    # (Multiplying by a boolean mask is Autograd safe!)
    va_final = va_new_full * bool_in_range
    vt_final = vt_new_full * bool_in_range
    vy_final = rp.zeros_like(va_final)

    # 6. Stack the arrays out-of-place to build the final (ctrl_pts, n_cp, 3) tensor
    rotor_V_wake_ind = rp.stack((va_final, vy_final, vt_final), axis=-1)

    return rotor_V_wake_ind