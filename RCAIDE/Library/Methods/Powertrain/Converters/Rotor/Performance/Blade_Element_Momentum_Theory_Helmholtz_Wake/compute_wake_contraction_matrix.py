# RCAIDE/Library/Methods/Powertrain/Converters/Rotor/Performance/BEMT_Hemholtz_Vortex_Theory/compute_wake_contraction_matrix.py
# 
# Created:  Sep 2020, M. Clarke 
#           Jul 2021, E. Botero
#           Sep 2021, R. Erhard

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ---------------------------------------------------------------------------------------------------------------------- 

# package imports
import RNUMPY as rp 

# ---------------------------------------------------------------------------------------------------------------------- 
#  compute_wake_contraction_matrix
# ---------------------------------------------------------------------------------------------------------------------- 
def compute_wake_contraction_matrix(prop, Nr, m, nts, X_pts, prop_outputs):
    """
    Computes slipstream development factor for all points along the wake slipstream.
    
    Parameters
    ----------
    prop : Data
        Propeller/rotor data structure with the following attributes:
            - radius_distribution : array_like
                Radial station positions [m]
            - number_of_blades : int
                Number of blades on the rotor
            - hub_radius : float
                Hub radius of the rotor [m]
            - tip_radius : float
                Tip radius of the rotor [m]
            - origin : array_like
                Origin coordinates of the rotor [m]
    Nr : int
        Number of radial stations on the propeller/rotor
    m : int
        Number of control points in segment
    nts : int
        Number of timesteps
    X_pts : array_like
        Location of wake points [m], shape (m, B, Nr, nts, 3)
        where B is number of blades
    prop_outputs : Data
        Propeller/rotor outputs with the following attributes:
            - disc_axial_induced_velocity : array_like
                Axial induced velocity on the disc [m/s]
            - velocity : array_like
                Velocity vector [m/s]
    
    Returns
    -------
    wake_contraction : array_like
        Wake contraction matrix, shape (m, B, Nr, nts)
        Ratio of contracted radius to original radius at each wake point
    
    Notes
    -----
    This function calculates the wake contraction factor for all points along the
    slipstream of a propeller or rotor. The wake contraction is a key factor in
    determining the induced velocities in the wake and the overall performance of
    the rotor.
    
    The computation follows these steps:
        1. Extract rotor geometry parameters (radius distribution, hub/tip radius)
        2. Calculate the distance of wake points from the rotor plane
        3. Compute the slipstream development factor based on distance
        4. Calculate the velocity ratio factor
        5. Compute the contracted radius at each wake point
        6. Calculate the wake contraction ratio (contracted radius / original radius)
    
    **Major Assumptions**
        * Fixed wake with helical shape
        * Wake contraction is primarily influenced by the axial induced velocity
        * The wake contraction model is based on momentum conservation
    
    **Theory**
    The wake contraction is modeled using a combination of distance-based and
    velocity-based factors. The slipstream development factor (s2) increases with
    distance from the rotor plane, while the velocity ratio factor (Kv) accounts
    for the effect of induced velocities on the wake contraction.
    
    The contracted radius at each wake point is calculated as:
    
    .. math::
        r'_{j+1} = \\sqrt{{r'_j}^2 + ({r_{j+1}}^2 - {r_j}^2) \\cdot K_v}
    
    where:
        - :math:`r'_j` is the contracted radius at station j
        - :math:`r_j` is the original radius at station j
        - :math:`K_v` is the velocity ratio factor
    
    References
    ----------
    [1] Stone, R. Hugh. "Aerodynamic modeling of the wing-propeller interaction for a tail-sitter unmanned air vehicle." Journal of Aircraft 45.1 (2008): 198-210.
    
    See Also
    --------
    RCAIDE.Library.Methods.Powertrain.Converters.Rotor.Performance.Blade_Element_Momentum_Theory_Helmholtz_Wake.wake_model
    """
    r                 = prop.radius_distribution  
    rdim              = Nr-1
    B                 = prop.number_of_blades
    va                = rp.mean(prop_outputs.disc_axial_induced_velocity, axis=2)  # induced velocitied averaged around the azimuth
    R0                = prop.hub_radius 
    R_p               = prop.tip_radius  
    s                 = X_pts[0,:,0,-1,:] - prop.origin[0][0]    #  ( control point, blade number,  location on blade, time step)  
    s2                = 1 + s/(rp.sqrt(s**2 + R_p**2))
    Kd                = rp.repeat(rp.atleast_2d(s2)[:, None, :], rdim , axis = 1)  
    
    # TO DO: UPDATE FOR ANGLES SO THAT VELOCITY IS COMPONENT IN ROTOR AXIS FRAME
    VX                = rp.repeat(rp.repeat(rp.atleast_2d(prop_outputs.velocity[:,0]).T, rdim, axis = 1)[:, :, None], nts , axis = 2) # dimension (num control points, propeller distribution, wake points )
   
    prop_dif          = rp.atleast_2d(va[:,1:] +  va[:,:-1])
    prop_dif          = rp.repeat(prop_dif[:,  :, None], nts, axis=2) 
     
    Kv                = (2*VX + prop_dif) /(2*VX + Kd*prop_dif)  
    
    r_diff            = rp.ones((m,rdim))*(r[1:]**2 - r[:-1]**2 )
    r_diff            = rp.repeat(rp.atleast_2d(r_diff)[:, :, None], nts, axis = 2) 
    r_prime           = rp.zeros((m,Nr,nts))                
    r_prime = r_prime.at[:,0,:].set(R0)
    for j in range(rdim):
        r_prime = r_prime.at[:,1+j,:].set(rp.sqrt(r_prime[:,j,:]**2 + (r_diff*Kv)[:,j,:]))
    
    wake_contraction  = rp.repeat((r_prime/rp.repeat(rp.atleast_2d(r)[:, :, None], nts, axis = 2))[:,None,:,:], B, axis = 1)            
    
    return wake_contraction 
            
