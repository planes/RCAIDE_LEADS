# RCAIDE/Methods/Aerodynamics/Common/Lift/BET_calculations.py
# 
# 
# Created:  Jul 2023, M. Clarke 

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------   
 
from RCAIDE.Framework.Core import interp2d 
from RCAIDE.Framework.Core. Data import Data

# package imports 
import RNUMPY as rp

# ----------------------------------------------------------------------------------------------------------------------
#  compute_airfoil_aerodynamics
# ----------------------------------------------------------------------------------------------------------------------   
def compute_airfoil_aerodynamics(beta,c,r,R,B,Wa,Wt,a,nu,airfoils,airfoil_locations,ctrl_pts,Nr,Na,tc,use_2d_analysis):
    """
    Cl, Cdval = compute_airfoil_aerodynamics( beta,c,r,R,B,
                                              Wa,Wt,a,nu,
                                              airfoils,a_loc
                                              ctrl_pts,Nr,Na,tc,use_2d_analysis )

    Computes the aerodynamic forces at sectional blade locations. If airfoil
    geometry and locations are specified, the forces are computed using the
    airfoil polar lift and drag surrogates, accounting for the local Reynolds
    number and local angle of attack.

    If the airfoils are not specified, an approximation is used.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       beta                       blade twist distribution                        [-]
       c                          chord distribution                              [-]
       r                          radius distribution                             [-]
       R                          tip radius                                      [-]
       B                          number of rotor blades                          [-]

       Wa                         axial velocity                                  [-]
       Wt                         tangential velocity                             [-]
       a                          speed of sound                                  [-]
       nu                         viscosity                                       [-]
       airfoil_data               Data structure of airfoil polar information     [-]
       ctrl_pts                   Number of control points                        [-]
       Nr                         Number of radial blade sections                 [-]
       Na                         Number of azimuthal blade stations              [-]
       tc                         Thickness to chord                              [-]
       use_2d_analysis            Specifies 2d disc vs. 1d single angle analysis  [Boolean]

    Outputs:
       Cl                       Lift Coefficients                         [-]
       Cdval                    Drag Coefficients  (before scaling)       [-]
       alpha                    section local angle of attack             [rad]

    """ 
    alpha    = beta - rp.arctan2(Wa,Wt)
    W        = (Wa*Wa + Wt*Wt)**0.5
    Ma       = W/a
    Re       = (W*c)/nu

    # If rotor airfoils are defined, use airfoil surrogate
    if len(airfoil_locations) != 0:
        a_loc = rp.array(airfoil_locations)
        # Compute blade Cl and Cd distribution from the airfoil data 
        if use_2d_analysis:
            # return the 2D Cl and CDval of shape (ctrl_pts, Nr, Na)
            Cl         = rp.zeros((ctrl_pts,Nr,Na))
            Cdval      = rp.zeros((ctrl_pts,Nr,Na))
            
            for jj,airfoil in enumerate(airfoils):
                pd                   = airfoil.polars
                Cl_af                = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.lift_coefficients) 
                Cdval_af             = interp2d(Re,alpha,pd.reynolds_numbers, pd.angle_of_attacks, pd.drag_coefficients)
                locs                 = rp.where(rp.array(a_loc) == jj )[0]
                Cl                   = Cl.at[:,locs,:].set(Cl_af[:,locs,:])
                Cdval                = Cdval.at[:,locs,:].set(Cdval_af[:,locs,:])
            alpha_disc           = alpha
            Re_disc              = Re
        else:
            # return the 1D Cl and CDval of shape (ctrl_pts, Nr)
            Cl         = rp.zeros((ctrl_pts,Nr))
            Cdval      = rp.zeros((ctrl_pts,Nr))             

            for jj, airfoil in enumerate(airfoils):
                pd = airfoil.polars

                # Interpolate Cl and Cd for this airfoil
                Cl_af    = interp2d(Re, alpha,
                                    pd.reynolds_numbers,
                                    pd.angle_of_attacks,
                                    pd.lift_coefficients)

                Cdval_af = interp2d(Re, alpha,
                                    pd.reynolds_numbers,
                                    pd.angle_of_attacks,
                                    pd.drag_coefficients)

                # Boolean mask for all blade stations using this airfoil
                mask = (rp.array(a_loc) == jj)

                # Scatter update into the correct columns
                Cl    = Cl.at[:, mask].set(Cl_af[:, mask])
                Cdval = Cdval.at[:, mask].set(Cdval_af[:, mask])


            alpha_disc = rp.tile(alpha[:,:, None], (1, 1, Na)) 
            Re_disc    = rp.tile(Re[:,:, None], (1, 1, Na))  

    else:
        # Estimate Cl max
        tc_1 = tc*100
        Cl_max_ref = -0.0009*tc_1**3 + 0.0217*tc_1**2 - 0.0442*tc_1 + 0.7005
        Cl_max_ref = rp.where(Cl_max_ref<0.7, 0.7, Cl_max_ref)
        Re_ref     = 9.*10**6
        Cl1maxp    = Cl_max_ref * ( Re / Re_ref ) **0.1

        # If not airfoil polar provided, use 2*pi as lift curve slope
        Cl = 2.*rp.pi*alpha

        # By 90 deg, it's totally stalled.
        Cl = rp.where(Cl>Cl1maxp, Cl1maxp, Cl)
        Cl = rp.where(alpha>=rp.pi/2, 0., Cl)

        # Scale for Mach, this is Karmen_Tsien
        # KT_cond = rp.logical_and((Ma[:,:]<1.),(Cl>0))
        # Cl[KT_cond] = Cl[KT_cond]/((1-Ma[KT_cond]*Ma[KT_cond])**0.5+((Ma[KT_cond]*Ma[KT_cond])/(1+(1-Ma[KT_cond]*Ma[KT_cond])**0.5))*Cl[KT_cond]/2)

        KT_cond = (Ma < 1.0) & (Cl > 0.0)

        Ma_KT = Ma[KT_cond]
        Cl_KT = Cl[KT_cond]

        root = rp.sqrt(1.0 - Ma_KT**2)
        den  = root + (Ma_KT**2)/(1.0 + root) * (Cl_KT/2.0)

        Cl = Cl.at[KT_cond].set(Cl_KT / den)


        # If the blade segments are supersonic, don't scale
        Cl[Ma[:,:]>=1.] = Cl[Ma[:,:]>=1.]

        #This is an atrocious fit of DAE51 data at RE=50k for Cd
        Cdval = (0.108*(Cl*Cl*Cl*Cl)-0.2612*(Cl*Cl*Cl)+0.181*(Cl*Cl)-0.0139*Cl+0.0278)*((50000./Re)**0.2)
        Cdval = rp.where(alpha>=rp.pi/2, 2., Cdval)
        
        alpha_disc = rp.tile(alpha[:,:, None], (1, 1, Nr)) 
        Re_disc    = rp.tile(Re[:,:, None], (1, 1, Nr))  

    # prevent zero Cl to keep Cd/Cl from breaking in BET
    Cl = rp.where(Cl==0,1e-6,Cl)

    return Cl, Cdval, alpha, alpha_disc,Ma,W,Re,Re_disc

# ----------------------------------------------------------------------------------------------------------------------
#  compute_inflow_and_tip_loss
# ----------------------------------------------------------------------------------------------------------------------    
def compute_inflow_and_tip_loss(r,R,Wa,Wt,B,et1=1,et2=1,et3=1):
    """
    Computes the inflow, lamdaw, and the tip loss factor, F.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
       r          radius distribution                                              [m]
       R          tip radius                                                       [m]
       Wa         axial velocity                                                   [m/s]
       Wt         tangential velocity                                              [m/s]
       B          number of rotor blades                                           [-]
       et1        tuning parameter for tip loss function 
       et2        tuning parameter for tip loss function 
       et3        tuning parameter for tip loss function 
       
    Outputs:               
       lamdaw     inflow ratio                                                     [-]
       F          tip loss factor                                                  [-]
       piece      output of a step in tip loss calculation (needed for residual)   [-]
    """
    lamdaw             = r*Wa/(R*rp.where(Wt==0, 1e-12, Wt))
    lamdaw             = rp.where(lamdaw<=0., 1e-12, lamdaw)

    tipfactor = B/2.0*(  (R/r)**et1 - 1  )**et2/lamdaw**et3 

    piece = rp.exp(-tipfactor)
    safe_piece = rp.where(piece>=1.0-1e-12, 1.0-1e-12, piece)
    Ftip  = 2.*rp.arccos(safe_piece)/rp.pi  

    return lamdaw, Ftip, piece