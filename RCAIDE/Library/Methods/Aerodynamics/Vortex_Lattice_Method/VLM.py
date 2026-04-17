
# VLM.py
# 
# Created: Aug 2025, M. Clarke    

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import RCAIDE
from RCAIDE.Framework.Core import Data 
from .compute_wing_induced_velocity      import compute_wing_induced_velocity
from .generate_vortex_distribution       import generate_vortex_distribution 
from .compute_RHS_matrix                 import compute_RHS_matrix

from RNUMPY.scipy.integrate import trapezoid
from copy import  deepcopy
import RNUMPY as rp
# ----------------------------------------------------------------------
#  Vortex Lattice
# ----------------------------------------------------------------------

def VLM(conditions,settings,geometry):
    """Uses the vortex lattice method to compute the lift, induced drag and moment coefficients.
     
    The user should be forwarned that this will cause very slight differences in results for 0 deflection due to
    the slightly different discretization.
    
    The user has the option to use the boundary conditions and induced velocities from either RCAIDE
    or VORLAX. See build_RHS in compute_RHS_matrix.py for more details.
    
    By default in Vortex_Lattice, VLM performs calculations based on panel coordinates with float32 precision. 
    The user may also choose to use float16 or float64, but be warned that the latter can be memory intensive.
    
    The user should note that fully capitalized variables correspond to a VORLAX variable of the same name
    
    
    Assumptions:
    The user provides either global discretezation (number_spanwise/chordwise_vortices) or
    separate discretization (wing/fuselage_spanwise/chordwise_vortices) in settings, not both.
    The set of settings not being used should be set to None.
    
    The VLM requires that the user provide a non-zero velocity that matches mach number. For
    surrogate training cases at mach 0, VLM uses a velocity of 1e-6 m/s

    
    Source:
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    
    Inputs:
    geometry.
       reference_area                          [m^2]
       wing.
         spans.projected                       [m]
         chords.root                           [m]
         chords.tip                            [m]
         sweeps.quarter_chord                  [radians]
         taper                                 [Unitless]
         twists.root                           [radians]
         twists.tip                            [radians]
         xz_plane_symmetric                    [Boolean]
         aspect_ratio                          [Unitless]
         areas.reference                       [m^2]
         vertical                              [Boolean]
         origin                                [m]
       fuselage.
        origin                                 [m]
        width                                  [m]
        heights.maximum                        [m]      
        lengths.nose                           [m]    
        lengths.tail                           [m]     
        lengths.total                          [m]     
        lengths.cabin                          [m]     
        fineness.nose                          [Unitless]
        fineness.tail                          [Unitless]
        
    settings.number_of_spanwise_vortices       [Unitless]  <---|
    settings.number_of_chordwise_vortices      [Unitless]  <---|
                                                               |--Either/or; see generate_vortex_distribution() for more details
    settings.wing_spanwise_vortices            [Unitless]  <---|
    settings.wing_chordwise_vortices           [Unitless]  <---|
    settings.fuselage_spanwise_vortices        [Unitless]  <---|
    settings.fuselage_chordwise_vortices       [Unitless]  <---|  
       
    settings.use_surrogate                     [Unitless]
    settings.propeller_wake_model              [Unitless] 
    settings.use_VORLAX_matrix_calculation     [boolean]
    settings.floating_point_precision          [float16/32/64]
       
    conditions.aerodynamics.angles.alpha       [radians]
    conditions.aerodynamics.angles.beta        [radians]
    conditions.freestream.mach_number          [Unitless]
    conditions.freestream.velocity             [m/s]
    conditions.static_stability.pitch_rate     [radians/s]
    conditions.static_stability.roll_rate      [radians/s]
    conditions.static_stability.yaw_rate       [radians/s]
       
    
    Outputs:    
    results.
        CL                                     [Unitless], CLTOT in VORLAX
        CDi                                    [Unitless], CDTOT in VORLAX
        CM                                     [Unitless], CMTOT in VORLAX
        CY                                     [Unitless], Total y force coeff
        CRTOT                                  [Unitless], Rolling moment coeff (unscaled)
        CL_mom                                 [Unitless], Rolling moment coeff (scaled by b_ref)
        CNTOT                                  [Unitless], Yawing  moment coeff (unscaled)
        CN                                     [Unitless], Yawing  moment coeff (scaled by b_ref)
        CL_wing                                [Unitless], CL  of each wing
        CDi_wing                               [Unitless], CDi of each wing
        cl_y                                   [Unitless], CL  of each strip
        cdi_y                                  [Unitless], CDi of each strip
        alpha_i                                [radians] , Induced angle of each strip in each wing (array of numpy arrays)
        CP                                     [Unitless], Pressure coefficient of each panel
        gamma                                  [Unitless], Vortex strengths of each panel

    
    Properties Used:
    N/A
    """
     
    S_ref = geometry.reference_area
    c_ref = geometry.reference_chord
    b_ref = geometry.reference_span  
    x_m   = geometry.mass_properties.center_of_gravity[0][0]
    z_m   = geometry.mass_properties.center_of_gravity[0][2] 

    # ---------------------------------------------------------------------------------------
    # Generate Panelization and Vortex Distribution
    # ------------------ -------------------------------------------------------------------- 
    VD                                                    = generate_vortex_distribution(conditions,settings,geometry) 
    settings.vortex_distribution.chord_lengths            = VD.chord_lengths[VD.leading_edge_indices != 0].reshape(len(VD.n_sw),rp.sum(VD.n_sw[0]))
    settings.vortex_distribution.n_sw                     = VD.n_sw 
    settings.vortex_distribution.n_cw                     = VD.n_cw 
    settings.vortex_distribution.n_w                      = VD.n_w 
    settings.vortex_distribution.chord_widths             = VD.chord_widths 
    settings.vortex_distribution.leading_edge_sweeps      = VD.leading_edge_sweeps 
    settings.vortex_distribution.XA1                      = VD.XA1
    settings.vortex_distribution.XA2                      = VD.XA2
    settings.vortex_distribution.XB1                      = VD.XB1
    settings.vortex_distribution.XB2                      = VD.XB2
    settings.vortex_distribution.YA1                      = VD.YA1
    settings.vortex_distribution.YA2                      = VD.YA2
    settings.vortex_distribution.YB1                      = VD.YB1
    settings.vortex_distribution.YB2                      = VD.YB2
    settings.vortex_distribution.ZA1                      = VD.ZA1
    settings.vortex_distribution.ZA2                      = VD.ZA2
    settings.vortex_distribution.ZB1                      = VD.ZB1
    settings.vortex_distribution.ZB2                      = VD.ZB2 
    settings.vortex_distribution.X                        = VD.X
    settings.vortex_distribution.Y                        = VD.Y
    settings.vortex_distribution.Z                        = VD.Z
    settings.vortex_distribution.XC                       = VD.XC
    settings.vortex_distribution.YC                       = VD.YC
    settings.vortex_distribution.ZC                       = VD.ZC
    settings.vortex_distribution.Y_SW                     = VD.Y_SW
     
    # unpack conditions--------------------------------------------------------------
    pwm      = settings.propeller_wake_model
    K_SPC    = settings.leading_edge_suction_multiplier 
    aoa      = conditions.aerodynamics.angles.alpha 
    mach     = conditions.freestream.mach_number 
    len_mach = len(mach)
    
    #For angular values, VORLAX uses degrees by default to radians via DTR (degrees to rads). 
    #RCAIDE uses radians and its Units system. All algular variables will be in radians or var*Units.degrees
    PSI       = conditions.aerodynamics.angles.beta    
    PITCHQ    = conditions.static_stability.pitch_rate              
    ROLLQ     = conditions.static_stability.roll_rate             
    YAWQ      = conditions.static_stability.yaw_rate 
    VINF      = conditions.freestream.velocity    
       
    #freestream 0 velocity safeguard
    if not conditions.freestream.velocity.all():
        if settings.use_surrogate:
            velocity                       = conditions.freestream.velocity
            velocity[velocity==0]          = rp.ones(len(velocity[velocity==0])) * 1e-6
            conditions.freestream.velocity = velocity
        else:
            raise AssertionError("VLM requires that conditions.freestream.velocity be specified and non-zero")    

    
    # Unpack vortex distribution 
    CHORD        = VD.chord_lengths
    chord_breaks = VD.chordwise_breaks
    span_breaks  = VD.spanwise_breaks
    RNMAX        = VD.panels_per_strip    
    LE_ind       = VD.leading_edge_indices != 0
    ZETA         = VD.tangent_incidence_angle
    RK           = VD.chordwise_panel_number
    
    exposed_leading_edge_flag = VD.exposed_leading_edge_flag
    
    YAH = VD.YAH*1.  
    YBH = VD.YBH*1. 
    XA1 = VD.XA1*1.
    XB1 = VD.XB1*1. 
    
    # Compute X and Z BAR ouside of generate_vortex_distribution to avoid requiring x_m and z_m as inputs 
    VD.XBAR = rp.ones(( len_mach,sum(LE_ind[0]))) * x_m 
    VD.ZBAR = rp.ones(( len_mach,sum(LE_ind[0]))) * z_m  
    
    # ---------------------------------------------------------------------------------------
    # STEP 10: Generate A and RHS matrices from VD and geometry
    # ------------------ --------------------------------------------------------------------    
    # Compute flow tangency conditions
    phi   = rp.arctan((VD.ZBC - VD.ZAC)/(VD.YBC - VD.YAC)) # dihedral angle 
    delta = rp.arctan((VD.ZC - VD.ZCH)/((VD.XC - VD.XCH))) # mean camber surface angle 

    # Build the RHS vector    
    rhs     = compute_RHS_matrix(VD,delta,phi,conditions,settings,geometry,pwm) 
    RHS     = rhs.RHS*1 # this matches numpy=1.26 in terms of dimension
    ONSET   = rhs.ONSET*1

    # Build induced velocity matrix, C_mn  
    C_mn, s, RFLAG, EW = compute_wing_induced_velocity(VD,mach,compute_EW=True)
    
    # Turn off sonic vortices when Mach>1
    RHS = RHS*RFLAG
    
    # To ensure compatibility for rp.linalg.solve across numpy1.0 and numpy2.0
    RHS = rp.atleast_3d(RHS)

    # Build Aerodynamic Influence Coefficient Matrix
    use_VORLAX_induced_velocity = settings.use_VORLAX_matrix_calculation
    if not use_VORLAX_induced_velocity:
        A =   rp.multiply(C_mn[:,:,:,0],rp.atleast_3d(rp.sin(delta)*rp.cos(phi))) \
            + rp.multiply(C_mn[:,:,:,1],rp.atleast_3d(rp.cos(delta)*rp.sin(phi))) \
            - rp.multiply(C_mn[:,:,:,2],rp.atleast_3d(rp.cos(phi)*rp.cos(delta)))   # validated from book eqn 7.42 
    else:
        A = EW

    # Compute vortex strength
    GAMMA  = rp.linalg.solve(A,RHS)

    # To ensure compatibility for rp.linalg.solve across numpy1.0 and numpy2.0
    RHS    = RHS.squeeze(axis=2)
    GAMMA  = GAMMA.squeeze(axis=2)

    # ---------------------------------------------------------------------------------------
    # STEP 11: Compute Pressure Coefficient
    # ------------------ --------------------------------------------------------------------   
    #VORLAX subroutine = PRESS
                  
    # spanwise strip exposure flag, always 0 for RCAIDE's infinitely thin airfoils. Needs to change if thick airfoils added
    RJTS = 0                         
    
    # COMPUTE FREE-STREAM AND ONSET FLOW PARAMETERS. Used throughout the remainder of VLM
    B2     = rp.tile((mach**2 - 1),int(VD.n_cp[0]))
    SINALF = rp.sin(aoa)
    COSALF = rp.cos(aoa)
    TANALF = rp.tan(aoa)
    SINPSI = rp.sin(PSI)
    COPSI  = rp.cos(PSI)
    COSIN  = COSALF *SINPSI *2.0
    COSINP = COSALF *SINPSI
    COSCOS = COSALF *COPSI
    PITCH  = PITCHQ /VINF
    ROLL   = ROLLQ /VINF
    YAW    = YAWQ /VINF    
    
    # reshape CHORD 
    dim_1 = len(rp.sum(LE_ind, axis=1))
    dim_2 = rp.sum(LE_ind, axis=1)[0]
    
    # COMPUTE EFFECT OF SIDESLIP on DCP intermediate variables. needs change if cosine chorwise spacing added
    FORAXL = COSCOS
    FORLAT = COSIN 
    TAN_LEi= (VD.XB1[:,LE_ind[0]]-VD.XA1[:,LE_ind[0]])/  rp.sqrt((VD.ZB1[:,LE_ind[0]]-VD.ZA1[:,LE_ind[0]])**2 +  (VD.YB1[:,LE_ind[0]]-VD.YA1[:,LE_ind[0]])**2)  
    TAN_TE = (VD.XB_TE - VD.XA_TE)/ rp.sqrt((VD.ZB_TE-VD.ZA_TE)**2 + (VD.YB_TE-VD.YA_TE)**2) 
    TAN_LE = rp.repeat( TAN_LEi, RNMAX[LE_ind].reshape(dim_1,dim_2)[0] , axis=1)
    
    TAN_LE = TAN_LE
    TNL    = TAN_LE * 1 # VORLAX's SIGN variable not needed, as these are taken directly from geometry
    TNT    = TAN_TE * 1
    XIA    = rp.broadcast_to((RK-1)/RNMAX, rp.shape(B2))
    XIB    = rp.broadcast_to((RK  )/RNMAX, rp.shape(B2))
    TANA   = TNL *(1. - XIA) + TNT *XIA
    TANB   = TNL *(1. - XIB) + TNT *XIB
    
    # cumsum GANT loop if KTOP > 0 (don't actually need KTOP with vectorized arrays and rp.roll)
    GFX    = VD.chord_lengths
    GANT   = strip_cumsum(GFX*GAMMA, chord_breaks[0], RNMAX[LE_ind].reshape(dim_1,dim_2)[0]  )
    GANT   = rp.roll(GANT,1)
    GANT[LE_ind] = 0 
    
    GLAT   = GANT *(TANA - TANB) - GFX *GAMMA *TANB
    cos_DL = (YBH-YAH)[LE_ind].reshape(dim_1,dim_2)/VD.D
    COS_DL = rp.repeat( cos_DL, RNMAX[LE_ind].reshape(dim_1,dim_2)[0] , axis=1)
    DCPSID = FORLAT * COS_DL *GLAT /(XIB - XIA)
    FACTOR = FORAXL + ONSET
    
    # COMPUTE LOAD COEFFICIENT
    GNET = GAMMA*FACTOR
    GNET = GNET *RNMAX /CHORD
    DCP  = 2*GNET + DCPSID
    CP   = DCP

    # ---------------------------------------------------------------------------------------
    # STEP 12: Compute aerodynamic coefficients 
    # ------------------ --------------------------------------------------------------------  
    # Flip coordinates on the other side of the wing
    boolean = YBH<0. 
    XA1[boolean], XB1[boolean] = XB1[boolean], XA1[boolean]
    YAH[boolean], YBH[boolean] = YBH[boolean], YAH[boolean]

    # Leading edge sweep. VORLAX does it panel by panel. This will be spanwise.
    TLE   = TAN_LE[LE_ind].reshape(dim_1,dim_2)
    B2_LE = B2[LE_ind].reshape(dim_1,dim_2)
    T2    = TLE*TLE
    STB   = rp.zeros_like(B2_LE)
    STB[B2_LE<T2] = rp.sqrt(T2[B2_LE<T2]-B2_LE[B2_LE<T2])
    
    # DL IS THE DIHEDRAL ANGLE (WITH RESPECT TO THE X-Y PLANE) OF
    # THE IR STREAMWISE STRIP OF HORSESHOE VORTICES. 
    COD = rp.cos(phi[LE_ind]).reshape(dim_1,dim_2)  # Just the LE values 
    SID = rp.sin(phi[LE_ind]).reshape(dim_1,dim_2)  # Just the LE values

    # Now on to each strip
    PION = 2.0 /RNMAX
    ADC  = 0.5*PION

    # XLE = LOCATION OF FIRST VORTEX MIDPOINT IN FRACTION OF CHORD.
    XLE = 0.125 *PION
    
    GAF = 0.5 + 0.5 *RJTS**2

    # CORMED IS LENGTH OF STRIP CENTERLINE BETWEEN LOAD POINT
    # AND TRAILING EDGE THIS PARAMETER IS USED IN THE COMPUTATION
    # OF THE STRIP ROLLING COUPLE CONTRIBUTION DUE TO SIDESLIP.
    X      = VD.XCH                       #x-coord of load point (horseshoe centroid)
    XTE    = (VD.XA_TE + VD.XB_TE)/2   #Trailing edge x-coord behind the control point  
    CORMED = XTE - X   

    # SINF REFERENCES THE LOAD CONTRIBUTION OF IRT-VORTEX TO THE
    # STRIP NOMINAL AREA, I.E., AREA OF STRIP ASSUMING CONSTANT
    # (CHORDWISE) HORSESHOE SPAN.    
    SINF = ADC * DCP # The horshoe span lengths have been removed since VST/VSS == 1 always

    # Split into chordwise strengths and sum into strips    
    # SICPLE = COUPLE (ABOUT STRIP CENTERLINE) DUE TO SIDESLIP.
    CNC    = rp.add.reduceat(SINF       ,chord_breaks[0],axis=1)
    SICPLE = rp.add.reduceat(SINF*CORMED,chord_breaks[0],axis=1)

    # COMPUTE SLOPE (TX) WITH RESPECT TO X-AXIS AT LOAD POINTS BY INTER
    # POLATING BETWEEN CONTROL POINTS AND TAKING INTO ACCOUNT THE LOCAL
    # INCIDENCE.    
    XX   = (RK - .75) *PION /2.0
    TX    = VD.SLOPE - ZETA
    CAXL  = -SINF*TX/(1.0+TX**2) # These are the axial forces on each panel
    BMLE  = (XLE-XX)*SINF        # These are moment on each panel
    
    # Sum onto the panel
    CAXL = rp.add.reduceat(CAXL,chord_breaks[0],axis=1)
    BMLE = rp.add.reduceat(BMLE,chord_breaks[0],axis=1)
    
    SICPLE = SICPLE * (-1) * COSIN * COD * GAF
    DCP_LE = DCP[LE_ind].reshape(dim_1,dim_2)
    
    # COMPUTE LEADING EDGE THRUST COEFF. (CSUC) BY CALCULATING
    # THE TOTAL INDUCED FLOW AT THE LEADING EDGE. THIS COMPUTATION
    # ONLY PERFORMED FOR COSINE CHORDWISE SPACING (LAX = 0).    
    # ** TO DO ** Add cosine spacing (earlier in VLM) to properly capture the magnitude of these earlier.
    # Right now, this computation still happens with linear spacing, though its effects are underestimated.
    CLE = compute_rotation_effects(VD, settings, EW, GAMMA, X, CHORD, XLE, VD.XBAR, rhs, COSINP, SINALF,COSCOS, PITCH, ROLL, YAW, STB, RNMAX)    
    
    # Leading edge suction multiplier. See documentation. This is a negative integer if used
    # Default to 1 unless specified otherwise
    SPC  = K_SPC*rp.ones_like(DCP_LE)
    
    # If the vehicle is subsonic and there is vortex lift enabled then SPC changes to -1
    VL   = rp.repeat(VD.vortex_lift,VD.n_sw[0], axis=1)
    m_b  = rp.atleast_2d(mach[:,0]<1.)
    SPC_cond      = VL*m_b.T
    SPC[SPC_cond] = -1.
    SPC           = SPC * exposed_leading_edge_flag
    
    CLE  = CLE + 0.5* DCP_LE *rp.sqrt(XLE[LE_ind].reshape(dim_1,dim_2))
    CSUC = 0.5*rp.pi*rp.abs(SPC)*(CLE**2)*STB 

    # TFX AND TFZ ARE THE COMPONENTS OF LEADING EDGE FORCE VECTOR ALONG
    # ALONG THE X AND Z BODY AXES.   
    
    SLE  = VD.SLOPE[LE_ind].reshape(dim_1,dim_2)
    ZETA = ZETA[LE_ind].reshape(dim_1,dim_2)
    XCOS = rp.cos(SLE-ZETA) 
    XSIN = rp.sin(SLE-ZETA) 
    TFX  =  1.*XCOS
    TFZ  = -1.*XSIN

    # If a negative number is used for SPC a different correction is used. See VORLAX documentation for Lan reference
    TFX[SPC<0] = XSIN[SPC<0]*rp.sign(DCP_LE)[SPC<0]
    TFZ[SPC<0] = rp.abs(XCOS)[SPC<0]*rp.sign(DCP_LE)[SPC<0]

    CAXL = CAXL - TFX*CSUC
    
    # Add a dimension into the suction to be chordwise
    CNC   = CNC + CSUC*rp.sqrt(1+T2)*TFZ
    
    # FCOS AND FSIN ARE THE COSINE AND SINE OF THE ANGLE BETWEEN
    # THE CHORDLINE OF THE IR-STRIP AND THE X-AXIS    
    FCOS = rp.cos(ZETA)
    FSIN = rp.sin(ZETA)
    
    # BFX, BFY, AND BFZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP FORCE CONTRIBUTION.
    BFX = -  CNC *FSIN + CAXL *FCOS
    BFY = - (CNC *FCOS + CAXL *FSIN) *SID
    BFZ =   (CNC *FCOS + CAXL *FSIN) *COD

    # CONVERT CNC FROM CN INTO CNC (COEFF. *CHORD).
    CHORD_strip = CHORD[LE_ind].reshape(dim_1,dim_2)   
    CNC         = CNC  * CHORD_strip
    BMLE        = BMLE * CHORD_strip

    # BMX, BMY, AND BMZ ARE THE COMPONENTS ALONG THE BODY AXES
    # OF THE STRIP MOMENT (ABOUT MOM. REF. POINT) CONTRIBUTION.
    X      = VD.XCH[LE_ind].reshape(dim_1,dim_2)  # These are all LE values
    Y      = VD.YCH[LE_ind].reshape(dim_1,dim_2)  # These are all LE values
    Z      = VD.ZCH[LE_ind].reshape(dim_1,dim_2)  # These are all LE values
    BMX    = BFZ * Y - BFY * (Z - VD.ZBAR)
    BMX    = BMX + SICPLE
    BMY    = BMLE * COD + BFX * (Z - VD.ZBAR) - BFZ * (X - VD.XBAR)
    BMZ    = BMLE * SID - BFX * Y + BFY * (X - VD.XBAR)
    CDC    = BFZ * SINALF +  (BFX *COPSI + BFY *SINPSI) * COSALF
    CDC    = CDC * CHORD_strip 

    ES     = 2*s[:,0,:][LE_ind].reshape(dim_1,dim_2)
    STRIP  = ES *CHORD_strip
    LIFT   = (BFZ *COSALF - (BFX *COPSI + BFY *SINPSI) *SINALF)*STRIP    
    MOMENT = STRIP * (BMY *COPSI - BMX *SINPSI)  
    FY     = (BFY *COPSI - BFX *SINPSI) *STRIP
    RM     = STRIP *(BMX *COSALF *COPSI + BMY *COSALF *SINPSI + BMZ *SINALF)
    YM     = STRIP *(BMZ *COSALF - (BMX *COPSI + BMY *SINPSI) *SINALF)

    # Lift coefficient
    Clift_y   = LIFT/CHORD_strip/ES  
    CL_wing   = rp.add.reduceat(LIFT,span_breaks[0],axis=1)/VD.wing_areas  
    CLift     = rp.atleast_2d(rp.sum(LIFT,axis=1)/S_ref).T          

    # Drag coefficient
    results   = compute_trefftz_plane_induced_drag(conditions, VD,Clift_y, X, Y, Z, CHORD_strip,S_ref,b_ref)       
    
    # force coefficeints 
    CX_for   = (TANALF * CLift -  results.CDrag_induced)/(COSALF - SINALF*TANALF)
    CZ_for   = (results.CDrag_induced+ CX_for*COSALF)/SINALF  
    CY_for   = rp.atleast_2d(rp.sum(FY,axis=1)/S_ref).T  

    # moment coefficients 
    CM_mom   = rp.atleast_2d(rp.sum(MOMENT,axis=1)/S_ref).T/c_ref  
    CL_mom   = rp.atleast_2d(rp.sum(RM,axis=1)/S_ref).T    /b_ref*(-1)                             
    CN_mom   = rp.atleast_2d(rp.sum(YM,axis=1)/S_ref).T    /b_ref*(-1)                            
   
    # ---------------------------------------------------------------------------------------
    # STEP 13: Pack outputs
    # ------------------ --------------------------------------------------------------------     
    results.CLift             = CLift  
    results.CX                = CX_for 
    results.CY                = CY_for  
    results.CZ                = -CZ_for 
    results.CL                = CL_mom 
    results.CM                = CM_mom  
    results.CN                = CN_mom  
    results.spanwise_stations = Y 
    results.CLift_wing        = CL_wing   
    results.sectional_CLift   = Clift_y     
    results.CP                = rp.array(CP    , dtype=settings.floating_point_precision )
    results.gamma             = rp.array(GAMMA , dtype=settings.floating_point_precision ) 
    results.V_distribution    = rhs.V_distribution
    results.V_x               = rhs.Vx_ind_total
    results.V_z               = rhs.Vz_ind_total 
 
    i = 0 
    dim_wing_lifts      = results.CLift_wing * VD.wing_areas
    dim_wing_drags      = results.CDrag_induced_wing * VD.wing_areas
    Clift_wings         = Data()
    Cdrag_wings         = Data()
    # Assign the lift and drag and non-dimensionalize
    for wing in geometry.wings.values():
        ref = wing.areas.reference
        if wing.xz_plane_symmetric:
            Clift_wings[wing.tag]      = rp.atleast_2d(rp.sum(dim_wing_lifts[:,i:(i+2)],axis=1)).T/ref
            Cdrag_wings[wing.tag]      = rp.atleast_2d(rp.sum(dim_wing_drags[:,i:(i+2)],axis=1)).T/ref
            i+=1
        else:
            Clift_wings[wing.tag]      = rp.atleast_2d(dim_wing_lifts[:,i]).T/ref
            Cdrag_wings[wing.tag]      = rp.atleast_2d(dim_wing_drags[:,i]).T/ref
        i+=1 
    results.CLift_wings         = Clift_wings
    results.CDrag_induced_wings = Cdrag_wings
    
    return results

# ----------------------------------------------------------------------
#  CLE rotation effects helper function
# ----------------------------------------------------------------------
def compute_rotation_effects(VD, settings, EW_large, GAMMA, X, CHORD, XLE, XBAR, 
                             rhs, COSINP, SINALF,COSCOS, PITCH, ROLL, YAW, STB, RNMAX):
    """ This computes the effects of the freestream and aircraft rotation rate on 
    CLE, the induced flow at the leading edge
    
    Assumptions:
    Several of the values needed in this calculation have been computed earlier and stored in VD
    
    Normally, VORLAX skips the calculation implemented in this function for linear 
    chordwise spacing (the if statement below). However, since the trends are correct, 
    albeit underestimated, this calculation is being forced here.    
    """
    LE_ind   = VD.leading_edge_indices != 0
    RNMAX    = VD.panels_per_strip
    dim_1    = len(rp.sum(LE_ind, axis=1))
    dim_2    = rp.sum(LE_ind, axis=1)[0]
    dim_3    = len(LE_ind[0])
    
    # Computate rotational effects (pitch, roll, yaw rates) on LE suction
    # pick leading edge strip values for EW and reshape GAMMA -> gamma accordingly
    EW    = EW_large[LE_ind, :].reshape(dim_1, dim_2, dim_3) 
    gamma = rp.array(rp.split(rp.repeat(GAMMA, dim_2, axis=0), dim_1))
    CLE   = (EW*gamma).sum(axis=2)
    
    # Up till EFFINC, some of the following values were computed in compute_RHS_matrix().
    #     EFFINC and ALOC are calculated the exact same way, except for the XGIRO term.
    # LOCATE VORTEX LATTICE CONTROL POINT WITH RESPECT TO THE
    # ROTATION CENTER (XBAR, 0, ZBAR). THE RELATIVE COORDINATES
    # ARE XGIRO, YGIRO, AND ZGIRO. 
    XGIRO = X - CHORD*XLE - rp.repeat( XBAR, RNMAX[LE_ind].reshape(dim_1,dim_2)[0] , axis=1) 
    YGIRO = rhs.YGIRO
    ZGIRO = rhs.ZGIRO
    
    # VX, VY, VZ ARE THE FLOW ONSET VELOCITY COMPONENTS AT THE LEADING
    # EDGE (STRIP MIDPOINT). VX, VY, VZ AND THE ROTATION RATES ARE
    # REFERENCED TO THE FREE STREAM VELOCITY.     
    VX = (COSCOS - PITCH*ZGIRO + YAW  *YGIRO)  
    VY = (COSINP - YAW  *XGIRO + ROLL *ZGIRO)  
    VZ = (SINALF - ROLL *YGIRO + PITCH*XGIRO)

    # CCNTL, SCNTL, SID, and COD were computed in compute_RHS_matrix()
    
    # EFFINC = COMPONENT OF ONSET FLOW ALONG NORMAL TO CAMBERLINE AT
    #          LEADING EDGE.
    EFFINC = VX *rhs.SCNTL + VY *rhs.CCNTL *rhs.SID - VZ *rhs.CCNTL *rhs.COD 
    CLE = CLE - EFFINC[LE_ind].reshape(dim_1,dim_2) 
    CLE = rp.where(STB > 0, CLE /RNMAX[LE_ind].reshape(dim_1,dim_2) /STB, CLE)
    
    return CLE

# ----------------------------------------------------------------------
#  Vectorized cumsum from indices
# ----------------------------------------------------------------------
def strip_cumsum(arr, chord_breaks, strip_lengths):
    """ Uses numpy to to compute a cumsum that resets along
    the leading edge of every strip.
    
    Assumptions:
    chordwise_breaks always starts at 0
    """    
    cumsum  = rp.cumsum(arr, axis=1)
    offsets = cumsum[:,chord_breaks-1]
    offsets = offsets.at[:,0].set(0)
    offsets = rp.repeat(offsets, strip_lengths, axis=1)
    return cumsum - offsets
    
    
def compute_trefftz_plane_induced_drag(conditions, VD, cl, x_dist, y_dist, z_dist, chord_dist,SREF,b_ref, v_inf=1):
     
    alpha   = conditions.aerodynamics.angles.alpha 
    n_cases = len(alpha) 
    n_wings = len(VD.n_sw[0])
    rho = 1
    
    # ------------------------------------------------------------------------------------------
    # Trefftz Plane Drag 
    # ------------------------------------------------------------------------------------------

    # Initialize results storage
    CDi_total         = rp.zeros(n_cases)
    CDi_wing          = rp.zeros((n_cases, n_wings))
    D_induced         = rp.zeros((n_cases, n_wings))
    Cd_i_distribution = rp.zeros_like(cl)
    alpha_i           = rp.zeros_like(cl) 

    # Calculate circulation for this case
    circulation_dist = 0.5 * chord_dist[0] * v_inf * cl 
   
    ws = 0
    # Induced velocity calculation for this case 
    for wing_index,wing_segments in enumerate(VD.n_sw[0]):
        ws_prev = ws*1
        ws += wing_segments
        circulation_segments = circulation_dist[:,ws_prev:ws]
        cl_segments = cl[:, ws_prev:ws]
        
        # Control points 
        y_control_points = y_dist[:,ws_prev:ws] 
        z_control_points = z_dist[:,ws_prev:ws] 
        x_control_points = x_dist[:,ws_prev:ws] 

        # Centerpoints 
        y_centerpoints = (y_control_points[:,:-1] + y_control_points[:,1:]) / 2
        z_centerpoints = (z_control_points[:,:-1] + z_control_points[:,1:]) / 2
        x_centerpoints = (x_control_points[:,:-1] + x_control_points[:,1:]) / 2

        # Shed vortex segments for this case
        differences = rp.diff(y_control_points,axis=1)
        direction   = rp.sign(differences) * rp.ones_like(y_centerpoints)
        shed_vortex_segments = direction * rp.diff(circulation_segments, axis=1)

        # Trefftz Plane Y-Z location:
        TP_y_centerpoints   = y_centerpoints
        TP_z_centerpoints   = rp.cos(alpha) * z_centerpoints - rp.sin(alpha) * x_centerpoints
        TP_y_control_points = y_control_points
        TP_z_control_points = rp.cos(alpha) * z_control_points - rp.sin(alpha) * x_control_points

        V_induced_list = []
        for j in range(len(y_control_points[0])): # Loop through each control point
            # Distance from segment to control point
            A = ( rp.tile(TP_y_control_points[:,j][:, None],(1,len(TP_y_centerpoints[0]) ))  - TP_y_centerpoints)**2
            B = ( rp.tile(TP_z_control_points[:,j][:, None],(1,len(TP_z_centerpoints[0]))) - TP_z_centerpoints)**2
            r = (A + B) ** (0.5)
            
            # Calculate normal vector to the wake trace
            if len(TP_y_control_points[0]) < 2 or len(TP_z_control_points[0]) < 2 : 
                v_ind_j = rp.zeros((n_cases,)) 
            else:
                # TODO: this slope should be calculated in relative to the spanwise direction
                # slope = rp.gradient(TP_z_control_points, TP_y_control_points[0],axis=1)

                dz = TP_z_control_points[:, 1:] - TP_z_control_points[:, :-1]
                dy = TP_y_control_points[0, 1:] - TP_y_control_points[0, :-1]

                # Replace zero spacing with 1.0 (or any constant)
                dy = rp.where(dy == 0, rp.ones_like(dy), dy)

                slope = dz / dy
                slope = rp.concatenate([slope[:, :1], slope], axis=1)
            
                # Normal vector to the wake trace
                phi_n = rp.arctan2(-1, slope[:,j])
                n_hat = rp.stack([rp.cos(phi_n), rp.sin(phi_n)], axis=1)
                
                # Calculate induced velocity vector
                v_hat_y = -1*( rp.tile(TP_z_control_points[:,j][:, None],(1,len(TP_z_centerpoints[0]))) - TP_z_centerpoints)/r
                v_hat_z =    ( rp.tile(TP_y_control_points[:,j][:, None],(1,len(TP_y_centerpoints[0]) ))  - TP_y_centerpoints)/r 
                
                v_y = v_hat_y * shed_vortex_segments / (2*rp.pi*r)
                v_z = v_hat_z * shed_vortex_segments / (2*rp.pi*r)

                # Downwash. Dot product of normal vector and induced velocity vector.
                v_ind_j = rp.sum(n_hat[:, 0][:, None] * v_y + n_hat[:, 1][:, None] * v_z, axis=1)
            
            V_induced_list.append(v_ind_j)
            
        V_induced = rp.stack(V_induced_list, axis=1)

        drag_sum = rp.sqrt(rp.square(y_control_points[:, 0]) + rp.square(z_control_points[:, 0]))
        s_wake   = rp.atleast_2d(deepcopy(drag_sum)).T
        for j in range(1,len(y_control_points[0])): 
            drag_sum +=  rp.sqrt(rp.square(y_control_points[:,j] - y_control_points[:,j-1]) + rp.square(z_control_points[:,j] - z_control_points[:,j-1]))
            s_wake    =  rp.hstack((s_wake, rp.atleast_2d(drag_sum).T))
        D_induced = D_induced.at[:,wing_index].set(-0.5 * rho * trapezoid(V_induced * circulation_segments, s_wake, axis=1))

        # Per-wing CDi (using wing's reference area)
        CDi_wing = CDi_wing.at[:,wing_index].set(D_induced[:,wing_index] / (0.5 * rho * v_inf**2 * VD.wing_areas[:,wing_index]))

        # Store results for this case
        alpha_i_case = rp.arctan(V_induced/ v_inf)
        Cd_i_distribution = Cd_i_distribution.at[:,ws_prev:ws].set(cl_segments * rp.sin(-alpha_i_case))
        alpha_i = alpha_i.at[:,ws_prev:ws].set(alpha_i_case)

    CDi_total = rp.sum(D_induced, axis=1) / (0.5 * rho * v_inf**2 * SREF) 
 
    # Package results
    results                          = Data()
    results.CDrag_induced            = CDi_total[:, rp.newaxis]
    results.sectional_CDrag_induced  = Cd_i_distribution
    results.CDrag_induced_wing       = CDi_wing
    results.alpha_induced            = alpha_i 
    return results