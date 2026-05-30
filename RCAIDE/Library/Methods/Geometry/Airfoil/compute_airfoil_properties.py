# RCAIDE/Library/Methods/Geometry/Two_Dimensional/Airfoil/compute_airfoil_properties.py
# 
# 
# Created:  Jul 2024, M. Clarke 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------    

# RCAIDE imports 
import RCAIDE
from RCAIDE.Framework.Core                                                          import Data , Units 
from RCAIDE.Library.Methods.Aerodynamics.Airfoil_Panel_Method.airfoil_analysis      import airfoil_analysis
from RCAIDE.Library.Methods.Geometry.Airfoil.import_airfoil_polars                  import import_airfoil_polars 
from RCAIDE.Library.Methods.Geometry.Airfoil.compute_naca_4series                   import compute_naca_4series  
from RCAIDE.Library.Methods.Aerodynamics.AERODAS.pre_stall_coefficients             import pre_stall_coefficients
from RCAIDE.Library.Methods.Aerodynamics.AERODAS.post_stall_coefficients            import post_stall_coefficients

# numpy imports 
import RNUMPY as rp
from scipy.interpolate     import RegularGridInterpolator
from RCAIDE.Framework.Core import interp2d 

# ----------------------------------------------------------------------------------------------------------------------
#  compute_airfoil_properties
# ----------------------------------------------------------------------------------------------------------------------   
def compute_airfoil_properties(airfoil_geometry, airfoil_polar_files = None,use_pre_stall_data=True):
    """This computes the aerodynamic properties and coefficients of an airfoil in stall regimes using pre-stall
    characterstics and AERODAS formation for post stall characteristics. This is useful for 
    obtaining a more accurate prediction of wing and blade loading as well as aeroacoustics. Pre stall characteristics 
    are obtained in the form of a text file of airfoil polar data obtained from airfoiltools.com
    
    Assumptions:
        None 
        
    Source
        None
        
    Inputs:
    airfoil_geometry                        <data_structure>
    airfoil_polar_files                     <string>
    boundary_layer_files                    <string>
    use_pre_stall_data                      [Boolean]
    Outputs:
    airfoil_data.
        cl_polars                           [unitless]
        cd_polars                           [unitless]      
        aoa_sweep                           [unitless]
         
    
    Properties Used:
    N/A
    """     
    Airfoil_Data   = Data()  
   
    # ----------------------------------------------------------------------------------------
    # Compute airfoil boundary layers properties 
    # ----------------------------------------------------------------------------------------   
    Airfoil_Data   = compute_boundary_layer_properties(airfoil_geometry,Airfoil_Data)
    num_polars     = len(Airfoil_Data.re_from_polar)
     
    # ----------------------------------------------------------------------------------------
    # Compute extended cl and cd polars 
    # ----------------------------------------------------------------------------------------    
    if airfoil_polar_files != None : 
        # check number of polars per airfoil in batch
        num_polars   = 0 
        n_p = len(airfoil_polar_files)
        if n_p < 3:
            raise AttributeError('Provide three or more airfoil polars to compute surrogate') 
        num_polars = max(num_polars, n_p)         
         
        # read in polars from files overwrite panel code  
        airfoil_file_data                          = import_airfoil_polars(airfoil_polar_files)  
        Airfoil_Data.aoa_from_polar                = airfoil_file_data.aoa_from_polar
        Airfoil_Data.re_from_polar                 = airfoil_file_data.re_from_polar 
        Airfoil_Data.lift_coefficients             = airfoil_file_data.lift_coefficients
        Airfoil_Data.drag_coefficients             = airfoil_file_data.drag_coefficients 
        
    # Get all of the coefficients for AERODAS wings
    AoA_sweep_deg         = rp.linspace(-14,90,105)
    AoA_sweep_rad         = AoA_sweep_deg*Units.degrees    
    
    # Create an infinite aspect ratio wing
    geometry              = RCAIDE.Library.Components.Wings.Wing()
    geometry.aspect_ratio = rp.inf
    geometry.section      = Data()  
    
    # AERODAS   
    CL = rp.zeros((num_polars,len(AoA_sweep_deg)))
    CD = rp.zeros((num_polars,len(AoA_sweep_deg)))  
    for j in range(num_polars):    
        airfoil_cl  = Airfoil_Data.lift_coefficients[j,:]
        airfoil_cd  = Airfoil_Data.drag_coefficients[j,:]   
        airfoil_aoa = Airfoil_Data.aoa_from_polar[j,:]/Units.degrees 
    
        # compute airfoil cl and cd for extended AoA range 
        CL_j, CD_j = compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg,geometry,use_pre_stall_data)  
        CL = CL.at[j,:].set(CL_j)
        CD = CD.at[j,:].set(CD_j) 
         
    # ----------------------------------------------------------------------------------------
    # Store data 
    # ----------------------------------------------------------------------------------------    
    Airfoil_Data.reynolds_numbers    = Airfoil_Data.re_from_polar
    Airfoil_Data.angle_of_attacks    = AoA_sweep_rad 
    Airfoil_Data.lift_coefficients   = CL 
    Airfoil_Data.drag_coefficients   = CD    
        
    return Airfoil_Data
 
def compute_extended_polars(airfoil_cl,airfoil_cd,airfoil_aoa,AoA_sweep_deg,geometry,use_pre_stall_data): 
    """ Computes the aerodynamic polars of an airfoil over an extended angle of attack range
    
    Assumptions:
    Uses AERODAS formulation for post stall characteristics 
    Source:
    Models of Lift and Drag Coefficients of Stalled and Unstalled Airfoils in Wind Turbines and Wind Tunnels
    by D Spera, 2008
        
    Inputs:
    airfoil_cl          [unitless]
    airfoil_cd          [unitless]
    airfoil_aoa         [degrees]
    AoA_sweep_deg       [unitless]
    geometry            [N/A]
    use_pre_stall_data  [boolean]
    Outputs: 
    CL                  [unitless]
    CD                  [unitless]       
    
    Properties Used:
    N/A
    """    
    
    # Create dummy settings and state
    settings                                              = Data()
    state                                                 = Data()
    state.conditions                                      = Data()
    state.conditions.aerodynamics                         = Data()
    state.conditions.aerodynamics.pre_stall_coefficients  = Data()
    state.conditions.aerodynamics.post_stall_coefficients = Data()  
    
    # # computing approximate zero lift aoa
    # AoA_sweep_radians    = AoA_sweep_deg*Units.degrees
    # airfoil_cl_plus      = airfoil_cl[airfoil_cl>0]
    # idx_zero_lift        = rp.where(airfoil_cl == min(airfoil_cl_plus))[0][0]
    # airfoil_cl_crossing  = airfoil_cl[idx_zero_lift-1:idx_zero_lift+1]
    # airfoil_aoa_crossing = airfoil_aoa[idx_zero_lift-1:idx_zero_lift+1]
    # try:
    #     A0  = rp.interp(0,airfoil_cl_crossing, airfoil_aoa_crossing)* Units.deg 
    # except:
    #     A0 = airfoil_aoa[idx_zero_lift] * Units.deg 


    # Sweep AoA in radians
    AoA_sweep_radians = AoA_sweep_deg * Units.degrees

    # Identify positive-lift indices
    pos_mask = airfoil_cl > 0
    pos_indices = rp.nonzero(pos_mask)[0]

    # If no positive-lift points exist, fall back to minimum Cl
    if pos_indices.size == 0:
        idx_zero_lift = rp.argmin(airfoil_cl)
    else:
        # Among positive Cl values, find the smallest positive Cl
        pos_cls = airfoil_cl[pos_indices]
        min_pos_cl = rp.min(pos_cls)
        idx_zero_lift = pos_indices[rp.argmin(rp.abs(pos_cls - min_pos_cl))]

    # Build a safe 2‑point bracket for interpolation
    i0 = rp.clip(idx_zero_lift - 1, 0, airfoil_cl.size - 1)
    i1 = rp.clip(idx_zero_lift + 1, 0, airfoil_cl.size - 1)

    cl_pair  = airfoil_cl[i0:i1+1]
    aoa_pair = airfoil_aoa[i0:i1+1]

    # Interpolate zero-lift AoA
    if cl_pair.size >= 2 and rp.all(~rp.isnan(cl_pair)):
        A0 = rp.interp(0.0, cl_pair, aoa_pair) * Units.deg
    else:
        A0 = airfoil_aoa[idx_zero_lift] * Units.deg


    # --- Max lift coefficient and associated AoA ---
    CL1max = rp.max(airfoil_cl)
    idx_aoa_max_prestall_cl = rp.argmin(rp.abs(airfoil_cl - CL1max))
    ACL1 = airfoil_aoa[idx_aoa_max_prestall_cl] * Units.degrees


    # --- Approximate lift curve slope (using AoA = 0° and 4°) ---
    # Find indices of AoA = 0 and AoA = 4
    idx_0 = rp.argmin(rp.abs(airfoil_aoa - 0.0))
    idx_4 = rp.argmin(rp.abs(airfoil_aoa - 4.0))

    cl_range  = airfoil_cl[rp.array([idx_0, idx_4])]
    aoa_range = airfoil_aoa[rp.array([idx_0, idx_4])] * Units.degrees

    S1 = (cl_range[1] - cl_range[0]) / (aoa_range[1] - aoa_range[0])


    # --- Max drag coefficient and associated AoA ---
    CD1max = rp.max(airfoil_cd)
    idx_aoa_max_prestall_cd = rp.argmin(rp.abs(airfoil_cd - CD1max))
    ACD1 = airfoil_aoa[idx_aoa_max_prestall_cd] * Units.degrees


    # Find the point of lowest drag and the CD
    idx_CD_min = rp.argmin(airfoil_cd)

    # Extract AoA and Cd at that point
    ACDmin = airfoil_aoa[idx_CD_min] * Units.degrees
    CDmin  = airfoil_cd[idx_CD_min]

    # Setup data structures for this run
    ones                                                      = rp.ones_like(AoA_sweep_radians)
    settings.section_zero_lift_angle_of_attack                = A0
    state.conditions.aerodynamics.angle_of_attack             = AoA_sweep_radians* ones  
    geometry.section.angle_attack_max_prestall_lift           = ACL1 * ones 
    geometry.pre_stall_maximum_drag_coefficient_angle         = ACD1 * ones 
    geometry.pre_stall_maximum_lift_coefficient               = CL1max * ones 
    geometry.pre_stall_maximum_lift_drag_coefficient          = CD1max * ones 
    geometry.section.minimum_drag_coefficient                 = CDmin * ones 
    geometry.section.minimum_drag_coefficient_angle_of_attack = ACDmin
    geometry.pre_stall_lift_curve_slope                       = S1
    
    # Get prestall coefficients
    CL1, CD1 = pre_stall_coefficients(state,settings,geometry)
    
    # Get poststall coefficents
    CL2, CD2 = post_stall_coefficients(state,settings,geometry)
    
    # Take the maxes
    CL_ij = rp.fmax(CL1,CL2)
    CL_ij = rp.where(AoA_sweep_radians <= A0, rp.fmin(CL1, CL2), CL_ij)
    
    CD_ij = rp.fmax(CD1,CD2)
    
    # Pack this loop
    CL = CL_ij
    CD = CD_ij
    
    if use_pre_stall_data == True:
        CL, CD = apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL, CD)
        
    return CL,CD

def apply_pre_stall_data(AoA_sweep_deg, airfoil_aoa, airfoil_cl, airfoil_cd, CL, CD): 
    '''Applies prestall data to lift and drag curve slopes
    
    Assumptions:
    None
    
    Source:
    None
    
    Inputs:
    AoA_sweep_deg  [degrees]
    airfoil_aoa    [degrees]
    airfoil_cl     [unitless]
    airfoil_cd     [unitless]
    CL             [unitless] 
    CD             [unitless]
    
    Outputs
    CL             [unitless]
    CD             [unitless] 
    
    
    Properties Used:
    N/A
    
    '''

    # # Coefficients in pre-stall regime taken from experimental data:
    # aoa_locs    = (AoA_sweep_deg>=airfoil_aoa[0]) * (AoA_sweep_deg<=airfoil_aoa[-1])
    # aoa_in_data = AoA_sweep_deg[aoa_locs]
    
    # # if the data is within experimental use it, if not keep the surrogate values
    # CL[aoa_locs] = airfoil_cl[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    # CD[aoa_locs] = airfoil_cd[abs(aoa_in_data[:,None] - airfoil_aoa[None,:]).argmin(axis=-1)]
    
    # # remove kinks/overlap between pre- and post-stall                
    # data_lb       = rp.where(CD == airfoil_cd[0])[0][0]
    # data_ub       = rp.where(CD == airfoil_cd[-1])[0][-1]
    # CD[0:data_lb] = rp.maximum(CD[0:data_lb], CD[data_lb]*rp.ones_like(CD[0:data_lb]))
    # CD[data_ub:]  = rp.maximum(CD[data_ub:],  CD[data_ub]*rp.ones_like(CD[data_ub:])) 

    # Identify AoA values that fall inside the experimental data range
    aoa_locs = (AoA_sweep_deg >= airfoil_aoa[0]) & (AoA_sweep_deg <= airfoil_aoa[-1])
    aoa_in_data = AoA_sweep_deg[aoa_locs]

    # For each AoA in the sweep, find the closest experimental AoA index
    nearest_idx = rp.argmin(rp.abs(aoa_in_data[:, None] - airfoil_aoa[None, :]), axis=-1)

    # Replace surrogate CL/CD with experimental values where available
    CL = CL.at[aoa_locs].set(airfoil_cl[nearest_idx])
    CD = CD.at[aoa_locs].set(airfoil_cd[nearest_idx])

    # ------------------------------------------------------------
    # Remove kinks / overlaps between pre‑ and post‑stall regions
    # ------------------------------------------------------------

    # Find lower bound: first index where CD equals the minimum experimental CD
    idx_lb = rp.argmin(rp.abs(CD - airfoil_cd[0]))

    # Find upper bound: last index where CD equals the maximum experimental CD
    # Create a boolean mask of matches
    matches = rp.abs(CD - airfoil_cd[-1]) < 1e-12

    # Multiply by indices to find the highest index where the condition is True
    idx_ub = rp.argmax(matches * rp.arange(len(CD)))

    # Enforce monotonicity before and after the experimental region
    CD = CD.at[:idx_lb].set(rp.maximum(CD[:idx_lb], CD[idx_lb]))
    CD = CD.at[idx_ub:].set(rp.maximum(CD[idx_ub:], CD[idx_ub]))

    
    return CL, CD
 
def compute_boundary_layer_properties(airfoil_geometry,Airfoil_Data): 
    '''Computes the boundary layer properties of an airfoil for a sweep of Reynolds numbers 
    and angle of attacks. 
    
    Source:
    None
    
    Assumptions:
    None 
    
    Inputs:
    airfoil_geometry   <data_structure>
    Airfoil_Data       <data_structure>
    
    Outputs:
    Airfoil_Data       <data_structure>
    
    Properties Used:
    N/A
    '''
    if airfoil_geometry == None:
        print('No airfoil defined, NACA 0012 surrogates will be used') 
        a_names                       = ['0012']                
        airfoil_geometry              = compute_naca_4series(a_names, npoints= 100)    
    
    AoA_sweep = rp.array([-4,0,2,4,8,10,14])*Units.degrees 
    Re_sweep  = rp.array([1,5,10,30,50,75,100])*1E4  
    AoA_vals  = rp.tile(AoA_sweep[None,:],(len(Re_sweep) ,1))
    Re_vals   = rp.tile(Re_sweep[:,None],(1, len(AoA_sweep)))     
    
    # run airfoil analysis  
    af_res  = airfoil_analysis(airfoil_geometry,AoA_vals,Re_vals) 
    
    # store data
    Airfoil_Data.aoa_from_polar                                     = rp.tile(AoA_sweep[None,:],(len(Re_sweep),1)) 
    Airfoil_Data.re_from_polar                                      = Re_sweep 
    Airfoil_Data.lift_coefficients                                  = af_res.cl_invisc
    Airfoil_Data.drag_coefficients                                  = af_res.cd_invisc
    Airfoil_Data.cm                                                 = af_res.cm_invisc
    Airfoil_Data.boundary_layer                                     = Data() 
    Airfoil_Data.boundary_layer.angle_of_attacks                    = AoA_sweep 
    Airfoil_Data.boundary_layer.reynolds_numbers                    = Re_sweep
    fL                                                              = rp.transpose(af_res.fL, axes=[1,2,0])
    fD                                                              = rp.transpose(af_res.fD, axes=[1,2,0]) 
    Airfoil_Data.lift_distribution_func                             = RegularGridInterpolator((AoA_sweep,Re_sweep),fL,method = 'linear',   bounds_error=False, fill_value=None)
    Airfoil_Data.drag_distribution_func                             = RegularGridInterpolator((AoA_sweep,Re_sweep),fD,method = 'linear',   bounds_error=False, fill_value=None)
    Airfoil_Data.lift_distribution                                  = fL 
    Airfoil_Data.drag_distribution                                  = fD  
    
    return Airfoil_Data