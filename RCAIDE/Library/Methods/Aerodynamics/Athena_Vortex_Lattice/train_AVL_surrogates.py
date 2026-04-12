# RCAIDE/Library/Methods/Aerodynamics/Vortex_Lattice_Method/train_AVL_surrogates.py
#  
# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE imports
import RCAIDE 
from RCAIDE.Framework.Mission.Common                                             import Results  
from RCAIDE.Library.Methods.Aerodynamics.Athena_Vortex_Lattice.run_AVL_analysis  import run_AVL_analysis  
 
# Package imports 
import os
import RNUMPY as rp
from shutil import rmtree    

# ----------------------------------------------------------------------------------------------------------------------
#  train_AVL_surrogates
# ---------------------------------------------------------------------------------------------------------------------- 
def train_AVL_surrogates(aerodynamics,vehicle):
    """Call methods to run VLM for sample point evaluation. 
    
    Assumptions:
        None
        
    Source:
        None

    Args:
        aerodynamics       : VLM analysis          [unitless] 
        
    Returns: 
        None    
    """ 
 
    run_folder             = os.path.abspath(aerodynamics.settings.filenames.run_folder) 
    training               = aerodynamics.training  
    AoA                    = training.angle_of_attack
    Mach                   = training.Mach
    side_slip_angle        = aerodynamics.settings.side_slip_angle
    roll_rate_coefficient  = aerodynamics.settings.roll_rate_coefficient
    pitch_rate_coefficient = aerodynamics.settings.pitch_rate_coefficient
    lift_coefficient       = aerodynamics.settings.lift_coefficient
    atmosphere             = RCAIDE.Framework.Analyses.Atmospheric.US_Standard_1976()
    atmo_data              = atmosphere.compute_values(altitude = 0.0)         
    
    len_AoA  = len(AoA)
    len_Mach = len(Mach)
    CM       = rp.zeros((len_AoA,len_Mach))
    CL       = rp.zeros_like(CM)
    CD       = rp.zeros_like(CM)
    e        = rp.zeros_like(CM)
    Cm_alpha = rp.zeros_like(CM)
    Cn_beta  = rp.zeros_like(CM)
    NP       = rp.zeros_like(CM)  

    # remove old files in run directory  
    if os.path.exists(aerodynamics.settings.filenames.run_folder):
        if aerodynamics.settings.new_regression_results:
            rmtree(run_folder)

    for i,_ in enumerate(Mach):
        # Set training conditions
        run_conditions = Results()
        run_conditions.expand_rows(len_AoA)
        run_conditions.aerodynamics.angles.alpha           = rp.array([AoA]).T  
        run_conditions.freestream.density                  = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*atmo_data.density 
        run_conditions.freestream.gravity                  = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*9.81          
        run_conditions.freestream.speed_of_sound           = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*atmo_data.speed_of_sound[0,0]  
        run_conditions.freestream.velocity                 = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*Mach[i] * run_conditions.freestream.speed_of_sound 
        run_conditions.freestream.mach_number              = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*Mach[i]
        run_conditions.aerodynamics.angles.beta            = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*side_slip_angle 
        run_conditions.static_stability.coefficients.roll  = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*roll_rate_coefficient   
        if lift_coefficient == None: 
            run_conditions.aerodynamics.coefficients.lift.inviscid.total= lift_coefficient
        else:
            run_conditions.aerodynamics.coefficients.lift.inviscid.total= rp.array([lift_coefficient]).T  
        run_conditions.static_stability.coefficients.pitch = rp.ones_like(run_conditions.aerodynamics.angles.alpha)*pitch_rate_coefficient 

        # Run Analysis at AoA[i] and Mach[i]
        run_AVL_analysis(aerodynamics,run_conditions, vehicle)
 
        CL = CL.at[:,i].set(run_conditions.aerodynamics.coefficients.lift.inviscid.total[:,0])
        CD = CD.at[:,i].set(run_conditions.aerodynamics.coefficients.drag.induced.total[:,0])
        e  = e .at[:,i].set(run_conditions.aerodynamics.coefficients.drag.induced.efficiency_factor[:,0])
        CM = CM.at[:,i].set(run_conditions.static_stability.coefficients.pitch[:,0])
        Cm_alpha = Cm_alpha.at[:,i].set(run_conditions.static_stability.derivatives.CM_alpha[:,0])
        Cn_beta = Cn_beta.at[:,i].set(run_conditions.static_stability.derivatives.CN_beta[:,0])
        NP = NP.at[:,i].set(run_conditions.static_stability.neutral_point[:,0])

    if aerodynamics.training_file:
        # load data 
        data_array   = rp.loadtxt(aerodynamics.training_file) 
        
        # convert from 1D to 2D        
        CL_1D         = rp.atleast_2d(data_array[:,0]) 
        CD_1D         = rp.atleast_2d(data_array[:,1])            
        e_1D          = rp.atleast_2d(data_array[:,2])
        CM_1D         = rp.atleast_2d(data_array[:,3]) 
        Cm_alpha_1D   = rp.atleast_2d(data_array[:,4])            
        Cn_beta_1D    = rp.atleast_2d(data_array[:,5])
        NP_1D         = rp.atleast_2d(data_array[:,6])

        # convert from 1D to 2D
        CL        = rp.reshape(CL_1D, (len_AoA,-1))
        CD        = rp.reshape(CD_1D, (len_AoA,-1))
        e         = rp.reshape(e_1D , (len_AoA,-1)) 
        CM        = rp.reshape(CM_1D, (len_AoA,-1))
        Cm_alpha  = rp.reshape(Cm_alpha_1D, (len_AoA,-1))
        Cn_beta   = rp.reshape(Cn_beta_1D , (len_AoA,-1))
        NP        = rp.reshape(NP_1D , (len_AoA,-1))

    # Save the data for regression 
    if aerodynamics.settings.new_regression_results:
        # convert from 2D to 1D
        CL_1D       = CL.reshape([len_AoA*len_Mach,1]) 
        CD_1D       = CD.reshape([len_AoA*len_Mach,1])  
        e_1D        = e.reshape([len_AoA*len_Mach,1]) 
        CM_1D       = CM.reshape([len_AoA*len_Mach,1]) 
        Cm_alpha_1D = Cm_alpha.reshape([len_AoA*len_Mach,1])  
        Cn_beta_1D  = Cn_beta.reshape([len_AoA*len_Mach,1])         
        NP_1D       = Cn_beta.reshape([len_AoA*len_Mach,1]) 
        rp.savetxt(vehicle.tag+'_stability_data.txt',rp.hstack([CL_1D,CD_1D,e_1D,CM_1D,Cm_alpha_1D, Cn_beta_1D,NP_1D ]),fmt='%10.8f',header='   CM       Cm_alpha       Cn_beta       NP ')

    # Store training data
    # Save the data for regression
    training_data = rp.zeros((7,len_AoA,len_Mach))
    training_data = training_data.at[0,:,:].set(CL)
    training_data = training_data.at[1,:,:].set(CD)
    training_data = training_data.at[2,:,:].set(e)
    training_data = training_data.at[3,:,:].set(CM)
    training_data = training_data.at[4,:,:].set(Cm_alpha)
    training_data = training_data.at[5,:,:].set(Cn_beta)
    training_data = training_data.at[6,:,:].set(NP)

    # Store training data
    training.coefficients = training_data
    
