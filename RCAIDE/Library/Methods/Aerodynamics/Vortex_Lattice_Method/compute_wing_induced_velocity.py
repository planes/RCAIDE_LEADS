# compute_wing_induced_velocity.py
# 
# Created:  Dec 2020, E. Botero
# Modified: May 2021, E. Botero  
#           Jun 2021, E. Botero  

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# package imports 
import RNUMPY as rp 

def compute_wing_induced_velocity(VD,mach,compute_EW=False):
    """ This computes the induced velocities at each control point of the vehicle vortex lattice 

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream
    
    Outside of a call to the VLM() function itself, EW does not need to be computed, as C_mn 
    provides the same information in the body-frame. 

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    VD       - vehicle vortex distribution                    [Unitless] 
    mach                                                      [Unitless] 
    
    Outputs:                                
    C_mn     - total induced velocity matrix                  [Unitless] 
    s        - semispan of the horshoe vortex                 [m] 
    t        - tangent of the horshoe vortex                  [-] 
    CHORD    - chord length for a panel                       [m] 
    RFLAG    - sonic vortex flag                              [boolean] 
    ZETA     - tangent incidence angle of the chordwise strip [-] 

    Properties Used:
    N/A
    """
    # unpack  
    LE_ind       = VD.leading_edge_indices != 0
    TE_ind       = VD.trailing_edge_indices != 0
    n_cp         = VD.n_cp
    n_mach       = len(mach)
    mach         = rp.array(mach,dtype=rp.float32)

    # Control points from the VLM 
    XAH   = rp.array(rp.atleast_2d(VD.XAH*1.),dtype=rp.float32)
    YAH   = rp.array(rp.atleast_2d(VD.YAH*1.),dtype=rp.float32)
    ZAH   = rp.array(rp.atleast_2d(VD.ZAH*1.),dtype=rp.float32)
    XBH   = rp.array(rp.atleast_2d(VD.XBH*1.),dtype=rp.float32)
    YBH   = rp.array(rp.atleast_2d(VD.YBH*1.),dtype=rp.float32)
    ZBH   = rp.array(rp.atleast_2d(VD.ZBH*1.),dtype=rp.float32)
    XA1   = rp.array(rp.atleast_2d(VD.XA1*1.),dtype=rp.float32)
    YA1   = rp.array(rp.atleast_2d(VD.YA1*1.),dtype=rp.float32)
    ZA1   = rp.array(rp.atleast_2d(VD.ZA1*1.),dtype=rp.float32)
    XB1   = rp.array(rp.atleast_2d(VD.XB1*1.),dtype=rp.float32)
    YB1   = rp.array(rp.atleast_2d(VD.YB1*1.),dtype=rp.float32)
    ZB1   = rp.array(rp.atleast_2d(VD.ZB1*1.),dtype=rp.float32)
    XA2   = rp.array(rp.atleast_2d(VD.XA2*1.),dtype=rp.float32)
    YA2   = rp.array(rp.atleast_2d(VD.YA2*1.),dtype=rp.float32)
    ZA2   = rp.array(rp.atleast_2d(VD.ZA2*1.),dtype=rp.float32)
    XB2   = rp.array(rp.atleast_2d(VD.XB2*1.),dtype=rp.float32)
    YB2   = rp.array(rp.atleast_2d(VD.YB2*1.),dtype=rp.float32)
    ZB2   = rp.array(rp.atleast_2d(VD.ZB2*1.),dtype=rp.float32)
    XC    = rp.array(rp.atleast_2d(VD.XC*1.),dtype=rp.float32)
    YC    = rp.array(rp.atleast_2d(VD.YC*1.),dtype=rp.float32)
    ZC    = rp.array(rp.atleast_2d(VD.ZC*1.),dtype=rp.float32)
    XA_TE = rp.array(rp.atleast_2d(VD.XA_TE*1.),dtype=rp.float32)
    XB_TE = rp.array(rp.atleast_2d(VD.XB_TE*1.),dtype=rp.float32)
    
    
    # Panel Dihedral Angle, using AH and BH location
    D      = rp.sqrt((YAH-YBH)**2+(ZAH-ZBH)**2)
    COS_DL = (YBH-YAH)/D    
    DL     = rp.arccos(COS_DL)
    DL     = rp.where(DL > rp.pi / 2, DL - rp.pi, DL) # This flips the dihedral angle for the other side of the wing
    
    # -------------------------------------------------------------------------------------------
    # Compute velocity induced by horseshoe vortex segments on every control point by every panel
    # ------------------------------------------------------------------------------------------- 
    # If YBH is negative, flip A and B, ie negative side of the airplane. Vortex order flips
    boolean = YAH>YBH
    # Define a quick functional swap helper
    def swap(cond, a, b):
        return rp.where(cond, b, a), rp.where(cond, a, b)

    # Execute the swaps, maintaining the clean visual columns
    XA1, XB1 = swap(boolean, XA1, XB1)
    YA1, YB1 = swap(boolean, YA1, YB1)
    ZA1, ZB1 = swap(boolean, ZA1, ZB1)
    
    XA2, XB2 = swap(boolean, XA2, XB2)
    YA2, YB2 = swap(boolean, YA2, YB2)
    ZA2, ZB2 = swap(boolean, ZA2, ZB2)
    
    XAH, XBH = swap(boolean, XAH, XBH)
    YAH, YBH = swap(boolean, YAH, YBH)
    ZAH, ZBH = swap(boolean, ZAH, ZBH)

    XA_TE, XB_TE = swap(boolean, XA_TE, XB_TE)
    
    # These vortices will use AH and BH, rather than the typical location
    xa = XAH[:,None,:]
    ya = YAH[:,None,:]
    za = ZAH[:,None,:]
    xb = XBH[:,None,:]
    yb = YBH[:,None,:]
    zb = ZBH[:,None,:]
    
    # This is not the control point for the panel, its the middle front of the vortex
    xc = 0.5*(xa+xb)
    yc = 0.5*(ya+yb)
    zc = 0.5*(za+zb)
    
    # This is the receiving point, or the control points
    xo = XC[:,:,None] 
    yo = YC[:,:,None] 
    zo = ZC[:,:,None] 
    
    # Incline the vortex
    theta    = rp.arctan2(zb-za,yb-ya)
    costheta = rp.cos(theta)
    sintheta = rp.sin(theta)
    
    # rotated axes
    x1bar = (xb - xc)
    y1bar = (yb - yc)*costheta + (zb - zc)*sintheta
    
    xobar = (xo - xc)
    yobar = (yo - yc)*costheta + (zo - zc)*sintheta
    zobar =-(yo - yc)*sintheta + (zo - zc)*costheta
    
    # COMPUTE COORDINATES OF RECEIVING POINT WITH RESPECT TO END POINTS OF SKEWED LEG.
    shape   = rp.shape(xobar)  
    s       = rp.repeat(rp.abs(y1bar),shape[1],axis=1)
    t       = rp.repeat(x1bar/y1bar ,shape[1],axis=1)
    
    X1 = xobar + t*s # In a planar case XC-XAH
    Y1 = yobar + s   # In a planar case YC-YAH
    X2 = xobar - t*s # In a planar case XC-XBH
    Y2 = yobar - s   # In a planar case YC-YBH
    
    # The cutoff hardcoded into vorlax
    CUTOFF = 0.8
    
    # CALCULATE AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG.
    XTY = xobar - t*yobar
    
    # The notation in this method is flipped from the paper
    B2 = rp.atleast_3d(mach**2-1.)
    
    # SET VALUES OF NUMERICAL TOLERANCE CONSTANTS.
    TOL    = s /500.0
    TOLSQ  = TOL *TOL
    TOLSQ2 = 2500.0 *TOLSQ
    ZSQ    = zobar *zobar
    YSQ1   = Y1 *Y1
    YSQ2   = Y2 *Y2
    RTV1   = YSQ1 + ZSQ
    RTV2   = YSQ2 + ZSQ
    XSQ1   = X1 *X1
    XSQ2   = X2 *X2
    
    # Split the vectors into subsonic and supersonic
    sub      = (B2<0)[:,0,0] 
    RO1      = B2*RTV1 
    RO2      = B2*RTV2 
    
    # ZERO-OUT PERTURBATION VELOCITY COMPONENTS
    U = rp.zeros((n_mach,shape[1],shape[2] ),dtype=rp.float32)
    V = rp.zeros((n_mach,shape[1],shape[2] ),dtype=rp.float32)
    W = rp.zeros((n_mach,shape[1],shape[2] ),dtype=rp.float32)    
    
    if rp.sum(sub)>0:
        # COMPUTATION FOR SUBSONIC HORSESHOE VORTEX
        U_sub, V_sub, W_sub = subsonic(zobar,XSQ1,RO1,XSQ2,RO2,XTY,t,B2,ZSQ,TOLSQ,X1,Y1,X2,Y2,RTV1,RTV2)   
        U, V, W = rp.where(sub[:, None, None], U_sub, U), rp.where(sub[:, None, None], V_sub, V), rp.where(sub[:, None, None], W_sub, W)
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX. some values computed in a preprocessing section in VLM
    sup = (B2>=0)[:,0,0]
    RFLAG = rp.ones((n_mach,shape[2]),dtype=rp.int8)
    if rp.sum(sup)>0:  
        RNMAX       = VD.panels_per_strip 
        CHORD       = rp.repeat(VD.chord_lengths[:, rp.newaxis, :],shape[1],axis=1)
        U_sup, V_sup, W_sup, RFLAG_sup  = supersonic(zobar,XSQ1,RO1,XSQ2,RO2,XTY,t,B2,ZSQ,TOLSQ,TOL,TOLSQ2,\
                                                    X1,Y1,X2,Y2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,n_cp,TE_ind,LE_ind)
        U, V, W = rp.where(sup[:, None, None], U_sup, U), rp.where(sup[:, None, None], V_sup, V), rp.where(sup[:, None, None], W_sup, W)
        RFLAG = rp.where(sup[:, None], RFLAG_sup, RFLAG)
    
    # Rotate into the vehicle frame and pack into a velocity matrix
    C_mn = rp.stack([U, V*costheta - W*sintheta, V*sintheta + W*costheta],axis=-1)
    
    
    if compute_EW == True:
        # Calculate the W velocity in the VORLAX frame for later calcs
        # The angles are Dihedral angle of the current panel - dihedral angle of the influencing panel
        COS1   = rp.cos(DL[:,:,None] - DL[:,None,:])
        SIN1   = rp.sin(DL[:,:,None] - DL[:,None,:]) 
        WEIGHT = 1
        
        EW = (W*COS1-V*SIN1)*WEIGHT
    else:
        # Assume that this function is being used outside of VLM, EW is not needed
        EW = rp.nan
        

    return C_mn, s, RFLAG, EW
    
def subsonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,X1,Y1,X2,Y2,RTV1,RTV2):
    """  This computes the induced velocities at each control point 
    of the vehicle vortex lattice for subsonic mach numbers

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    Z       Z relative location of the vortices          [m]
    XSQ1    X1 squared                                   [m^2]
    RO1     coefficient                                  [-]
    XSQ2    X2 squared                                   [m^2]
    RO2     coefficient                                  [-]
    XTY     AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    T       tangent of the horshoe vortex                [-] 
    B2      mach^2-1 (-beta2)                            [-] 
    ZSQ     Z squared                                    [m^2] 
    TOLSQ   coefficient                                  [-]
    X1      X coordinate of the left side of the vortex  [m]
    Y1      Y coordinate of the left side of the vortex  [m]
    X2      X coordinate of the right side of the vortex [m]
    Y2      Y coordinate of the right side of the vortex [m]
    RTV1    coefficient                                  [-]
    RTV2    coefficient                                  [-]

    
    Outputs:           
    U       X velocity       [unitless]
    V       Y velocity       [unitless]
    W       Z velocity       [unitless]

    Properties Used:
    N/A
    """  
    
    CPI  = 4 * rp.pi
    RAD1 = rp.sqrt(XSQ1 - RO1)
    RAD2 = rp.sqrt(XSQ2 - RO2)
    
    TBZ  = (T*T-B2)*ZSQ
    DENOM = XTY * XTY + TBZ
    
    TOLSQ = rp.broadcast_to(TOLSQ,rp.shape(DENOM))
    
    DENOM = rp.where(DENOM < TOLSQ, TOLSQ, DENOM)
    
    FB1 = (T *X1 - B2 *Y1) /RAD1
    FT1 = (X1 + RAD1) /(RAD1 *RTV1)
    FT1 = rp.where(RTV1 < TOLSQ, 0.0, FT1)
    
    FB2 = (T *X2 - B2 *Y2) /RAD2
    FT2 = (X2 + RAD2) /(RAD2 *RTV2)
    FT2 = rp.where(RTV2 < TOLSQ, 0.0, FT2)
    
    QB = (FB1 - FB2) /DENOM
    ZETAPI = Z /CPI
    U = ZETAPI *QB
    U = rp.where(ZSQ < TOLSQ, 0.0, U)
    V = ZETAPI * (FT1 - FT2 - QB *T)
    V = rp.where(ZSQ < TOLSQ, 0.0, V)
    W = - (QB *XTY + FT1 *Y1 - FT2 *Y2) /CPI
    
    return U, V, W

def supersonic(Z,XSQ1,RO1,XSQ2,RO2,XTY,T,B2,ZSQ,TOLSQ,TOL,TOLSQ2,X1,Y1,X2,Y2,RTV1,RTV2,CUTOFF,CHORD,RNMAX,n_cp,TE_ind, LE_ind):
    """  This computes the induced velocities at each control point 
    of the vehicle vortex lattice for supersonic mach numbers

    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    Z            Z relative location of the vortices          [m]
    XSQ1         X1 squared                                   [m^2]
    RO1          coefficient                                  [-]
    XSQ2         X2 squared                                   [m^2]
    RO2          coefficient                                  [-]
    XTY          AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    T            tangent of the horshoe vortex                [-] 
    B2           mach^2-1 (-beta2)                            [-] 
    ZSQ          Z squared                                    [m^2] 
    TOLSQ        coefficient                                  [-]
    X1           X coordinate of the left side of the vortex  [m]
    Y1           Y coordinate of the left side of the vortex  [m]
    X2           X coordinate of the right side of the vortex [m]
    Y2           Y coordinate of the right side of the vortex [m]
    RTV1         coefficient                                  [-]
    RTV2         coefficient                                  [-]
    CUTOFF       coefficient                                  [-]
    CHORD        chord length for a panel                     [m] 
    RNMAX        number of chordwise panels                   [-]
    n_cp         number of control points                     [-]
    TE_ind       indices of the trailing edge                 [-]
    LE_ind       indices of the leading edge                  [-]
    

    
    Outputs:           
    U       X velocity        [unitless]
    V       Y velocity        [unitless]
    W       Z velocity        [unitless]
    RFLAG   sonic vortex flag [boolean] 

    Properties Used:
    N/A
    """      
    
    CPI    = 2 * rp.pi
    T2     = T*T
    ZETAPI = Z/CPI
    shape  = rp.shape(RO1)
    RAD1   = rp.sqrt(XSQ1 - RO1)
    RAD2   = rp.sqrt(XSQ2 - RO2)
    
    RAD1 = rp.where(rp.isnan(RAD1), 0.0, RAD1)
    RAD2 = rp.where(rp.isnan(RAD2), 0.0, RAD2)
    

    DENOM = XTY * XTY + (T2 - B2) * ZSQ
    SIGN  = rp.where(DENOM < 0, -1., 1.)
    TOLSQ = rp.broadcast_to(TOLSQ, shape)
    
    DENOM_COND = rp.abs(DENOM) < TOLSQ
    DENOM = rp.where(DENOM_COND, SIGN * TOLSQ, DENOM)
    

    REPS = CUTOFF * XSQ1
    RAD1 = rp.where(X1 < TOL, 0.0, RAD1)
    
    # Create a boolean for various conditions for F1 that goes to zero
    bool1 = ~((X1 < TOL) | (RAD1 == 0.) | (RO1 > REPS))
    FRAD1 = RAD1 
    
    # Protect against Div-by-Zero
    safe_FRAD1 = rp.where(FRAD1 == 0., 1.0, FRAD1)
    safe_RTV1  = rp.where(RTV1 == 0.,  1.0, RTV1)
    
    FB1 = (T * X1 - B2 * Y1) / safe_FRAD1
    FT1 = X1 / (safe_FRAD1 * safe_RTV1)
    FT1 = rp.where(RTV1 < TOLSQ, 0., FT1)
    
    # Use the boolean to turn things off
    FB1 = rp.where(~rp.isfinite(FB1), 1., FB1)
    FT1 = rp.where(~rp.isfinite(FT1), 1., FT1)    
    
    # Apply the boolean mask
    FB1 = rp.where(bool1, FB1, 0.)
    FT1 = rp.where(bool1, FT1, 0.)
    

    REPS = CUTOFF * XSQ2
    RAD2 = rp.where(X2 < TOL, 0.0, RAD2)
    
    # Round 2
    # Create a boolean for various conditions for F2 that goes to zero
    bool2 = ~((X2 < TOL) | (RAD2 == 0.) | (RO2 > REPS))
    FRAD2 = RAD2    
    
    safe_FRAD2 = rp.where(FRAD2 == 0., 1.0, FRAD2)
    safe_RTV2  = rp.where(RTV2 == 0.,  1.0, RTV2)
    
    FB2 = (T * X2 - B2 * Y2) / safe_FRAD2
    FT2 = X2 / (safe_FRAD2 * safe_RTV2)
    FT2 = rp.where(RTV2 < TOLSQ, 0., FT2)
    
    # Use the boolean to turn things off
    FB2 = rp.where(~rp.isfinite(FB2), 1., FB2)
    FT2 = rp.where(~rp.isfinite(FT2), 1., FT2)    
    
    FB2 = rp.where(bool2, FB2, 0.)
    FT2 = rp.where(bool2, FT2, 0.)
    

    QB = (FB1 - FB2) / DENOM
    U  = ZETAPI * QB
    V  = ZETAPI * (FT1 - FT2 - QB * T)
    W  = - (QB * XTY + FT1 * Y1 - FT2 * Y2) / CPI    
    
    # COMPUTATION FOR SUPERSONIC HORSESHOE VORTEX WHEN RECEIVING POINT IS IN THE PLANE OF THE HORSESHOE
    in_plane = rp.broadcast_to(ZSQ < TOLSQ2, shape)
    

    W_in_full = supersonic_in_plane(RAD1, RAD2, Y1, Y2, TOL, XTY, CPI)

    U = rp.where(in_plane, 0., U)
    V = rp.where(in_plane, 0., V)
    W = rp.where(in_plane, W_in_full, W)
    
    # DETERMINE IF TRANSVERSE VORTEX LEG OF HORSESHOE ASSOCIATED TO THE
    # CONTROL POINT UNDER CONSIDERATION IS SONIC (SWEPT PARALLEL TO MACH
    # LINE)? IF SO THEN RFLAG = 0.0, OTHERWISE RFLAG = 1.0.
    size   = shape[1]
    n_mach = shape[0]    
    T2S    = T2[:, 0, :] 
    T2F    = rp.zeros((n_mach, size))
    T2A    = rp.zeros((n_mach, size))
    
    # Setup masks
    F_mask = rp.ones((n_mach, size), dtype=bool).at[TE_ind].set(False)
    A_mask = rp.ones((n_mach, size), dtype=bool).at[LE_ind].set(False)
    
    # Apply the mask
    T2F = T2F.at[A_mask].set(T2S[F_mask])
    T2A = T2A.at[F_mask].set(T2S[A_mask])
    
    # Zero out terms on the LE and TE
    T2F = T2F.at[TE_ind].set(0.)
    T2A = T2A.at[LE_ind].set(0.)

    TRANS = (B2[:, :, 0] - T2F) * (B2[:, :, 0] - T2A)
    
    # RFLAG and FLAG_bool
    RFLAG = rp.where(TRANS < 0, 0, 1).astype(rp.int8)
    FLAG_bool = rp.where(TRANS < 0, True, False).reshape((n_mach, size, -1))
    

    # COMPUTE THE GENERALIZED PRINCIPAL PART OF THE VORTEX-INDUCED VELOCITY INTEGRAL, WWAVE.
    # FROM LINE 2647 VORLAX, the IR .NE. IRR means that we're looking at vortices that affect themselves
    COX = CHORD / RNMAX[:, :, None] 
    
    # Differentiate the variable names so we don't overwrite the original T2 and B2
    T2_full  = rp.broadcast_to(T2, shape) * rp.eye(n_cp[0, 0], dtype=rp.int8)
    B2_full  = rp.broadcast_to(B2, shape) * rp.eye(n_cp[0, 0], dtype=rp.int8)
    COX_full = rp.broadcast_to(COX, shape) * rp.eye(n_cp[0, 0], dtype=rp.int8)
    
    WWAVE_mask = B2_full > T2_full
    

    safe_radicand = rp.where(WWAVE_mask, B2_full - T2_full, 0.0)
    safe_COX      = rp.where(WWAVE_mask & (COX_full != 0.), COX_full, 1.0)
    
    WWAVE_calc = -0.5 * rp.sqrt(safe_radicand) / safe_COX
    WWAVE = rp.where(WWAVE_mask, WWAVE_calc, 0.0)

    W = W + WWAVE    
    
    # IF CONTROL POINT BELONGS TO A SONIC HORSESHOE VORTEX, AND THE
    # SENDING ELEMENT IS SUCH HORSESHOE, THEN MODIFY THE NORMALWASH
    # COEFFICIENTS IN SUCH A WAY THAT THE STRENGTH OF THE SONIC VORTEX
    # WILL BE THE AVERAGE OF THE STRENGTHS OF THE HORSESHOES IMMEDIATELY
    # IN FRONT OF AND BEHIND IT.
    
    # Zero out the row
    FLAG_bool_rep     = rp.broadcast_to(FLAG_bool,shape)
    W = rp.where(FLAG_bool_rep, 0.0, W) # Default to zero

    # # The self velocity goes to 2
    # FLAG_bool_split   = rp.array(rp.split(FLAG_bool.ravel(),n_mach))
    # FLAG_ind          = rp.array(rp.where(FLAG_bool_split))
    # squares           = rp.zeros((size,size,n_mach))
    # squares[FLAG_ind[1],FLAG_ind[1],FLAG_ind[0]] = 1
    # squares           = rp.ravel(squares,order='F')
    
    # FLAG_bool_self    = rp.where(squares==1)[0]
    # W                 = W.ravel()
    # W[FLAG_bool_self] = 2. # It's own value, -2
    
    # # The panels before and after go to -1
    # FLAG_bool_bef = FLAG_bool_self - 1
    # FLAG_bool_aft = FLAG_bool_self + 1
    # W[FLAG_bool_bef] = -1.
    # W[FLAG_bool_aft] = -1.
    
    # W = rp.reshape(W,shape)

    # Split boolean mask into n_mach chunks
    FLAG_bool_split = rp.array(rp.split(FLAG_bool.ravel(), n_mach))

    # JAX-compatible replacement for np.where(FLAG_bool_split)
    # Returns (mach_idx, panel_idx)
    mach_idx, panel_idx = rp.nonzero(FLAG_bool_split)

    # Allocate squares
    squares = rp.zeros((size, size, n_mach))

    # JAX-compatible indexed assignment
    squares = squares.at[panel_idx, panel_idx, mach_idx].set(1)

    # Flatten in Fortran order
    squares = rp.ravel(squares, order="F")

    # JAX-compatible replacement for np.where(squares == 1)
    FLAG_bool_self = rp.nonzero(squares == 1)[0]

    # Update W
    W = W.ravel()
    W = W.at[FLAG_bool_self].set(2.0)

    # Panels before and after → -1
    FLAG_bool_bef = FLAG_bool_self - 1
    FLAG_bool_aft = FLAG_bool_self + 1

    W = W.at[FLAG_bool_bef].set(-1.0)
    W = W.at[FLAG_bool_aft].set(-1.0)

    # Restore original shape
    W = rp.reshape(W, shape)


    return U, V, W, RFLAG


def supersonic_in_plane(RAD1,RAD2,Y1,Y2,TOL,XTY,CPI):
    """  This computes the induced velocities at each control point 
    in the special case where the vortices lie in the same plane
    
    Assumptions: 
    Trailing vortex legs infinity are alligned to freestream
    In plane vortices only produce W velocity

    Source:  
    1. Miranda, Luis R., Robert D. Elliot, and William M. Baker. "A generalized vortex 
    lattice method for subsonic and supersonic flow applications." (1977). (NASA CR)
    
    2. VORLAX Source Code

    Inputs: 
    RAD1    array of zeros                               [-]
    RAD2    array of zeros                               [-]
    Y1      Y coordinate of the left side of the vortex  [m]
    Y2      Y coordinate of the right side of the vortex [m]
    TOL     coefficient                                  [-]
    XTY     AXIAL DISTANCE BETWEEN PROJECTION OF RECEIVING POINT ONTO HORSESHOE PLANE AND EXTENSION OF SKEWED LEG [m]
    CPI     2 Pi                                         [radians]

    
    Outputs:           
    W       Z velocity       [unitless]

    Properties Used:
    N/A
    """    
    
    cond1 = rp.abs(Y1) > TOL
    cond2 = rp.abs(Y2) > TOL
    condW = rp.abs(XTY) > TOL
    
    safe_Y1  = rp.where(cond1, Y1, 1.0)
    safe_Y2  = rp.where(cond2, Y2, 1.0)
    safe_XTY = rp.where(condW, XTY, 1.0)
    
    F1 = rp.where(cond1, RAD1 / safe_Y1, 0.0)
    F2 = rp.where(cond2, RAD2 / safe_Y2, 0.0)
    
    W  = rp.where(condW, (-F1 + F2) / (safe_XTY * CPI), 0.0)

    return W
