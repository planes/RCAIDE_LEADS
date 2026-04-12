# RCAIDE/Methods/Aeroacoustics/Physics_Based/Rotor/compute_rotor_noise.py
# 
# 
# Created:  Jul 2023, M. Clarke  

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE Imports 
from RCAIDE.Framework.Core import  Data   
from RCAIDE.Library.Methods.Aeroacoustics.Common.decibel_arithmetic                                  import SPL_arithmetic  
from RCAIDE.Library.Methods.Aeroacoustics.Common.compute_noise_source_coordinates                    import compute_rotor_point_source_coordinates  
from RCAIDE.Library.Methods.Aeroacoustics.Physics_Based_Frequency_Domain.Rotor.harmonic_noise_point  import harmonic_noise_point
from RCAIDE.Library.Methods.Aeroacoustics.Physics_Based_Frequency_Domain.Rotor.harmonic_noise_line   import harmonic_noise_line
from RCAIDE.Library.Methods.Aeroacoustics.Physics_Based_Frequency_Domain.Rotor.harmonic_noise_plane  import harmonic_noise_plane 
from RCAIDE.Library.Methods.Aeroacoustics.Physics_Based_Frequency_Domain.Rotor.broadband_noise       import broadband_noise
from RCAIDE.Library.Methods.Aeroacoustics.Common                                                     import atmospheric_attenuation
from RCAIDE.Library.Methods.Aeroacoustics.Metrics.A_weighting_metric                                 import A_weighting_metric  
from RCAIDE.Library.Methods.Geometry.Airfoil.import_airfoil_geometry                                 import import_airfoil_geometry
from RCAIDE.Library.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis                       import airfoil_analysis

# Python package imports   
import RNUMPY as rp    
from RCAIDE.Framework.Core import interp2d 

# ----------------------------------------------------------------------------------------------------------------------    
#  Rotor Noise 
# ----------------------------------------------------------------------------------------------------------------------    
def compute_rotor_noise(microphone_locations,rotor,segment,settings, rotor_index = 0, previous_rotor_tag = None, identical_propulsors=True):
    ''' This is a collection medium-fidelity frequency domain methods for rotor acoustic noise prediction which 
    computes the acoustic signature (sound pressure level, weighted sound pressure levels,
    and frequency spectrums of a system of rotating blades           
        
    Assumptions:
    None

    Source:
    None
    
    Inputs:
        rotors                  - data structure of rotors                            [None]
        segment                 - flight segment data structure                       [None] 
        results                 - data structure containing of acoustic data          [None]
        settings                - accoustic settings                                  [None]
                               
    Outputs:
        Results.    
            blade_passing_frequencies      - blade passing frequencies                           [Hz]
            SPL                            - total SPL                                           [dB]
            SPL_dBA                        - dbA-Weighted SPL                                    [dBA]
            SPL_1_3_spectrum               - 1/3 octave band spectrum of SPL                     [dB]
            SPL_1_3_spectrum_dBA           - 1/3 octave band spectrum of A-weighted SPL          [dBA]
            SPL_broadband_1_3_spectrum     - 1/3 octave band broadband contribution to total SPL [dB] 
            SPL_harmonic_1_3_spectrum      - 1/3 octave band harmonic contribution to total SPL  [dB]
            SPL_harmonic_bpf_spectrum_dBA  - A-weighted blade passing freqency spectrum of 
                                             harmonic compoment of SPL                           [dB]
            SPL_harmonic_bpf_spectrum      - blade passing freqency spectrum of harmonic
                                             compoment of SPL                                    [dB] 
     
    Properties Used:
        N/A   
    '''
 
    # unpack 
    conditions           = segment.state.conditions 
    num_mic              = len(microphone_locations[:,0]) 
    num_cpt              = conditions._size
    num_f                = len(settings.center_frequencies)
      
    # create data structures for computation
    aeroacoustics   = Data()  
    Results     = Data()

    Results.SPL                                           = rp.zeros((num_cpt,num_mic))
    Results.SPL_dBA                                       = rp.zeros_like(Results.SPL)
    Results.SPL_harmonic                                  = rp.zeros_like(Results.SPL)
    Results.SPL_broadband                                 = rp.zeros_like(Results.SPL)
    Results.blade_passing_frequencies                     = rp.zeros(num_f)
    Results.SPL_1_3_spectrum                              = rp.zeros((num_cpt,num_mic,num_f)) 
    Results.SPL_harmonic_bpf_spectrum                     = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_harmonic_bpf_spectrum_dBA                 = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.one_third_frequency_spectrum                  = settings.center_frequencies 
    Results.SPL_1_3_spectrum                              = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_1_3_spectrum_dBA                          = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_harmonic_1_3_spectrum                     = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_harmonic_1_3_spectrum_dBA                 = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_broadband_1_3_spectrum                    = rp.zeros_like(Results.SPL_1_3_spectrum)
    Results.SPL_broadband_1_3_spectrum_dBA                = rp.zeros_like(Results.SPL_1_3_spectrum)

    # compute position vector from point source (or should it be origin) at rotor hub to microphones 
    coordinates   = compute_rotor_point_source_coordinates(rotor,conditions,microphone_locations,settings)        

    for cpt in range(num_cpt): 
        # ----------------------------------------------------------------------------------
        # Harmonic Noise
        # ---------------------------------------------------------------------------------- 
        # harmonic noise with planar load distribution
        if settings.fidelity == 'plane_source': 
            aeroacoustic_data = segment.state.conditions.energy.converters[rotor.tag]       
            Re                = aeroacoustic_data.disc_reynolds_number
            AOA_sec           = aeroacoustic_data.disc_effective_angle_of_attack  
            a_loc             = rotor.airfoil_polar_stations
            num_az            = aeroacoustic_data.number_azimuthal_stations     
            airfoils          = rotor.airfoils         
            for jj,airfoil in enumerate(airfoils):
                airfoil_points      = airfoil.number_of_points 
            chord_coord             = int(rp.floor(airfoil_points/2))       
                
            if (identical_propulsors == False) and rotor_index !=0: 
                prev_aeroacoustic_data                   = segment.state.conditions.energy.converters[previous_rotor_tag]                 
                prev_aeroacoustic_data                   = segment.state.conditions.energy.converters[rotor.tag]  
                aeroacoustic_data.disc_lift_distribution = prev_aeroacoustic_data.disc_lift_distribution
                aeroacoustic_data.disc_drag_distribution = prev_aeroacoustic_data.disc_lift_distribution
                aeroacoustic_data.disc_lift_coefficient  = prev_aeroacoustic_data.disc_lift_coefficient 
                aeroacoustic_data.disc_drag_coefficient  = prev_aeroacoustic_data.disc_drag_coefficient  
                aeroacoustic_data.blade_upper_surface    = prev_aeroacoustic_data.blade_upper_surface
                aeroacoustic_data.blade_lower_surface    = prev_aeroacoustic_data.blade_lower_surface
            else: 
                # Lift and Drag - coefficients and distributions 
                fL      = rp.tile(rp.zeros_like(Re)[:,:,:,None],(1,1,1,chord_coord))
                fD      = rp.zeros_like(fL)
                CL      = rp.zeros_like(Re)
                CD      = rp.zeros_like(Re) 
                y_up    = rp.zeros_like(fL)
                y_low   = rp.zeros_like(fL)
                                  
                # 1. Determine static dimensions and extract FULL rows (no slicing by locs!)
                num_locs    = len(a_loc)
                a_loc_array = rp.array(a_loc)
                
                alpha_full  = rp.atleast_2d(AOA_sec[cpt, :, :].flatten())
                Re_full     = rp.atleast_2d(Re[cpt, :, :].flatten())      

                for jj, airfoil in enumerate(airfoils):    
                    # 2. Build a static boolean mask for this specific airfoil
                    mask = (a_loc_array == jj)
                    
                    # Expand the mask to match the 3D and 4D array shapes so it broadcasts safely
                    mask_3d = mask[None, :, None]       # Shape: (1, num_locs, num_az)
                    mask_4d = mask[None, :, None, None] # Shape: (1, num_locs, num_az, chord_coord)
                    
                    pd = airfoil.polars 
                    
                    # 3. Evaluate the physics for the ENTIRE array at once
                    if settings.use_plane_loading_surrogate: 
                        fL_full   = pd.lift_distribution_func((alpha_full, Re_full)).reshape(1, num_locs, num_az, chord_coord)
                        fD_full   = pd.drag_distribution_func((alpha_full, Re_full)).reshape(1, num_locs, num_az, chord_coord)
                        cl_invisc = interp2d(Re_full, alpha_full, pd.reynolds_numbers, pd.angle_of_attacks, pd.lift_coefficients)
                        cd_visc   = interp2d(Re_full, alpha_full, pd.reynolds_numbers, pd.angle_of_attacks, pd.drag_coefficients)   
                        CL_full   = cl_invisc.reshape(1, num_locs, num_az)
                        CD_full   = cd_visc.reshape(1, num_locs, num_az)
                        
                    else: 
                        airfoil_geometry   = import_airfoil_geometry(airfoil.coordinate_file, airfoil_points)
                        airfoil_properties = airfoil_analysis(airfoil_geometry, alpha_full, Re_full)
                        fL_full = airfoil_properties.fL.reshape(chord_coord, num_locs, num_az, 1).swapaxes(0, 3)
                        fD_full = airfoil_properties.fD.reshape(chord_coord, num_locs, num_az, 1).swapaxes(0, 3)
                        CL_full = airfoil_properties.cl_invisc.reshape(1, num_locs, num_az)
                        CD_full = airfoil_properties.cd_visc.reshape(1, num_locs, num_az)
                        
                    # Geometry arrays (assuming they broadcast natively)
                    y_up_full  = airfoil.geometry.y_upper_surface
                    y_low_full = airfoil.geometry.y_lower_surface

                    # 4. Use out-of-place assignment (.at[].set) combined with rp.where() to safely merge
                    # Note: We use cpt:cpt+1 instead of cpt to maintain the 4D/3D shape for the mask to broadcast against
                    fL = fL.at[cpt:cpt+1, :, :, :].set(rp.where(mask_4d, fL_full, fL[cpt:cpt+1, :, :, :]))
                    fD = fD.at[cpt:cpt+1, :, :, :].set(rp.where(mask_4d, fD_full, fD[cpt:cpt+1, :, :, :]))
                    
                    CL = CL.at[cpt:cpt+1, :, :].set(rp.where(mask_3d, CL_full, CL[cpt:cpt+1, :, :]))
                    CD = CD.at[cpt:cpt+1, :, :].set(rp.where(mask_3d, CD_full, CD[cpt:cpt+1, :, :]))
                    
                    y_up  = y_up.at[cpt:cpt+1, :, :, :].set(rp.where(mask_4d, y_up_full, y_up[cpt:cpt+1, :, :, :]))
                    y_low = y_low.at[cpt:cpt+1, :, :, :].set(rp.where(mask_4d, y_low_full, y_low[cpt:cpt+1, :, :, :]))
                        
                aeroacoustic_data.disc_lift_distribution = fL
                aeroacoustic_data.disc_drag_distribution = fD
                aeroacoustic_data.disc_lift_coefficient  = CL
                aeroacoustic_data.disc_drag_coefficient  = CD 
                aeroacoustic_data.blade_upper_surface    = y_up
                aeroacoustic_data.blade_lower_surface    = y_low                        
                        
            harmonic_noise_plane(conditions,coordinates,rotor,settings,aeroacoustics,cpt)
        elif settings.fidelity == 'line_source': 
            harmonic_noise_line(conditions,coordinates,rotor,settings,aeroacoustics,cpt)
        else:
            harmonic_noise_point(conditions,coordinates,rotor,settings,aeroacoustics,cpt) 
    
        # ----------------------------------------------------------------------------------    
        # Broadband Noise
        # ---------------------------------------------------------------------------------- 
        broadband_noise(conditions,coordinates,rotor,settings,aeroacoustics,cpt)  
    
        # ----------------------------------------------------------------------------------    
        # Atmospheric attenuation 
        # ----------------------------------------------------------------------------------
        delta_atmo = atmospheric_attenuation(rp.linalg.norm(coordinates.X_r[:,0,0,0,:],axis=1),settings.center_frequencies)
    
        # ----------------------------------------------------------------------------------    
        # Combine Harmonic (periodic/tonal) and Broadband Noise
        # ----------------------------------------------------------------------------------
        num_mic      = len(coordinates.X_hub[0,:,0,0])
        SPL_total_1_3_spectrum      = 10*rp.log10( 10**(aeroacoustics.SPL_prop_harmonic_1_3_spectrum/10) + 10**(aeroacoustics.SPL_prop_broadband_1_3_spectrum/10)) - rp.tile(delta_atmo[cpt,None,:],(1,num_mic,1))  
        SPL_total_1_3_spectrum[rp.isnan(SPL_total_1_3_spectrum)] = 0 
    
        # ----------------------------------------------------------------------------------
        # Summation of spectra from propellers into one SPL and store results
        # ----------------------------------------------------------------------------------
        Results.SPL = Results.SPL.at[cpt,:].set(SPL_arithmetic(SPL_total_1_3_spectrum[0], sum_axis=1))
        Results.SPL_dBA = Results.SPL_dBA.at[cpt,:].set(SPL_arithmetic(A_weighting_metric(SPL_total_1_3_spectrum[0],settings.center_frequencies), sum_axis=1))
        Results.SPL_harmonic = Results.SPL_harmonic.at[cpt,:].set(SPL_arithmetic(aeroacoustics.SPL_prop_harmonic_1_3_spectrum[0], sum_axis=1))
        Results.SPL_broadband = Results.SPL_broadband.at[cpt,:].set(SPL_arithmetic(aeroacoustics.SPL_prop_broadband_1_3_spectrum[0], sum_axis=1))
          
        # blade passing frequency         
        Results.SPL_harmonic_bpf_spectrum = Results.SPL_harmonic_bpf_spectrum.at[cpt,:,:].set(aeroacoustics.SPL_prop_harmonic_bpf_spectrum)
        Results.SPL_harmonic_bpf_spectrum_dBA = Results.SPL_harmonic_bpf_spectrum_dBA.at[cpt,:,:].set(A_weighting_metric(Results.SPL_harmonic_bpf_spectrum[cpt,:,:],aeroacoustics.f))
          
        # 1/3 octave band   
        Results.SPL_1_3_spectrum = Results.SPL_1_3_spectrum.at[cpt,:,:].set(SPL_total_1_3_spectrum)
        Results.SPL_1_3_spectrum_dBA = Results.SPL_1_3_spectrum_dBA.at[cpt,:,:].set(A_weighting_metric(Results.SPL_1_3_spectrum[cpt,:,:],settings.center_frequencies))
        Results.SPL_harmonic_1_3_spectrum = Results.SPL_harmonic_1_3_spectrum.at[cpt,:,:].set(aeroacoustics.SPL_prop_harmonic_1_3_spectrum)
        Results.SPL_harmonic_1_3_spectrum_dBA = Results.SPL_harmonic_1_3_spectrum_dBA.at[cpt,:,:].set(A_weighting_metric(Results.SPL_harmonic_1_3_spectrum[cpt,:,:],settings.center_frequencies))
        Results.SPL_broadband_1_3_spectrum = Results.SPL_broadband_1_3_spectrum.at[cpt,:,:].set(aeroacoustics.SPL_prop_broadband_1_3_spectrum)
        Results.SPL_broadband_1_3_spectrum_dBA = Results.SPL_broadband_1_3_spectrum_dBA.at[cpt,:,:].set(A_weighting_metric(Results.SPL_broadband_1_3_spectrum[cpt,:,:],settings.center_frequencies))
    
    # A-weighted
    conditions.aeroacoustics.converters[rotor.tag] = Results 
    return rotor.tag 
