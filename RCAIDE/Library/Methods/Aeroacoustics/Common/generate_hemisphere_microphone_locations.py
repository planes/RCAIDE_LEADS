# RCAIDE/Methods/Aeroacoustics/Common/generate_microphone_locations.py
# 
# 
# Created:  Oct 2023, A. Molloy  

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ---------------------------------------------------------------------------------------------------------------------- 

# package imports 
import RNUMPY as rp

# ---------------------------------------------------------------------------------------------------------------------- 
#  generate_hemisphere_microphone_locations
# ---------------------------------------------------------------------------------------------------------------------- 
def generate_hemisphere_microphone_locations(settings): 
    """This computes the microphones locations in a noise hemisphere
            
    Assumptions:
        None

    Source:
        N/A  

    Inputs:   
    settings.
    
        r     - noise hemisphere radius                   [meters]
        n     - noise hemisphere microphone resolution    [unitless]
        phi   - noise hemisphere phi angle bounds         [radians]
        theta - noise hemisphere theta angle bounds       [radians] 
    
    Outputs: 
        gm_mic_locations   - cartesian coordiates of all microphones defined  [meters] 
    
    Properties Used:
        N/A       
    """     
    r     = settings.noise_hemisphere_radius                  
    phi   = settings.noise_hemisphere_phi_angles   
    theta = settings.noise_hemisphere_theta_angles   
 
    x     = r * rp.outer(rp.sin(phi), rp.cos(theta))
    y     = r * rp.outer(rp.sin(phi), rp.sin(theta))
    z     = r * rp.outer(rp.cos(phi), rp.ones(rp.size(theta))) 
 
    num_gm                = len(z.flatten())
    gm_mic_locations      = rp.zeros((num_gm,3))  
    gm_mic_locations = gm_mic_locations.at[:,0].set(x.flatten())
    gm_mic_locations = gm_mic_locations.at[:,1].set(y.flatten())
    gm_mic_locations = gm_mic_locations.at[:,2].set(z.flatten())
    
    return gm_mic_locations   