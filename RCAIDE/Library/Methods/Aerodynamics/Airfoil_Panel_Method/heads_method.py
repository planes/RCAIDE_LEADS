# RCAIDE/Methods/Aerodynamics/Airfoil_Panel_Method/heads_method.py
# 
# 
# Created:  Mar 2024, Niranjan Nanjappa

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE imports    
from RCAIDE.Framework.Core import Data 

# package imports  
import RNUMPY as rp 

# ----------------------------------------------------------------------------------------------------------------------
# heads_method
# ---------------------------------------------------------------------------------------------------------------------- 
def heads_method(npanel,ncases,ncpts,NU,DEL_0,THETA_0,DELTA_STAR_0,CF_0,ShapeFactor_0,RE_L,TURBULENT_COORD,VE_I,DVE_I,TURBULENT_SURF,tol,wrong_columns):
    """ Computes the boundary layer characteristics in turbulent
    flow pressure gradients

    Source:
    Head, M. R., and P. Bandyopadhyay. "New aspects of turbulent boundary-layer structure."
    Journal of fluid mechanics 107 (1981): 297-338.

    Assumptions:
    None  

    Inputs: 
    ncases         - number of cases                                                               [unitless]
    ncpts          - number of control points                                                      [unitless]
    DEL_0          - intital bounday layer thickness                                               [m]
    DELTA_STAR_0   - initial displacement thickness                                                [m]
    CF_0           - initial value of the skin friction coefficient                                [unitless]
    H_0            - initial value of the shape factor                                             [unitless]
    THETA_0        - initial momentum thickness                                                    [m]
    TURBULENT_SURF - normalized length of surface                                                  [unitless]
    RE_L           - Reynolds number                                                               [unitless]
    TURBULENT_COORD- x coordinate on surface of airfoil                                            [unitless] 
    VE_I           - boundary layer velocity at all panels                                         [m/s-m] 
    DVE_I          - derivative of boundary layer velocity at all panels                           [m/s^2] 
    npanel         - number of points on surface                                                   [unitless]
    tol            - boundary layer error correction tolerance                                     [unitless]

    Outputs: 
    RESULTS.
      X_H          - reshaped distance along airfoil surface                    [unitless]
      THETA_H      - momentum thickness                                         [m]
      DELTA_STAR_H - displacement thickness                                     [m] 
      H_H          - shape factor                                               [unitless]
      CF_H         - friction coefficient                                       [unitless]
      RE_THETA_H   - Reynolds number as a function of momentum thickness        [unitless]
      RE_X_H       - Reynolds number as a function of distance                  [unitless]
      DELTA_H      - boundary layer thickness                                   [m]
       
    Properties Used:
    N/A
    """    
    # Initialize vectors 
    X_H          = rp.zeros((npanel,ncases,ncpts))
    THETA_H      = rp.zeros_like(X_H)
    DELTA_STAR_H = rp.zeros_like(X_H)
    H_H          = rp.zeros_like(X_H)
    CF_H         = rp.zeros_like(X_H) 
    RE_THETA_H   = rp.zeros_like(X_H)
    RE_X_H       = rp.zeros_like(X_H)
    DELTA_H      = rp.zeros_like(X_H)      
    
    for case in range(ncases):
        for cpt in range(ncpts):  
            # length of tubulent surface  
            l = TURBULENT_SURF[case,cpt] 
            if l == 0.0:
                pass
            else: 
                def getcf(ind, nu, H, THETA):
                    ReTheta = Ve_i[ind]*THETA/nu;
                    cf_var = 0.246*(10**(-0.678*H))*(ReTheta**-0.268);
                    return cf_var
                
                def getH(H1_var):
                    if H1_var<3.3:
                        H_var = 3.0    # This is the bug. Why does the code keep defaulting to this scenario (Does this indicate stall ?)
                    elif H1_var < 5.39142:
                        H_var = 0.6778 + 1.153793*(H1_var-3.3)**-0.32637;
                    elif H1_var >= 5.39142:
                        H_var = 1.1 + 0.8598636*(H1_var - 3.3)**-0.777;
                    return H_var 
             
                # define RK4 slope function for Theta
                def dTheta_by_dx(index, X, THETA, VETHETAH1):
                    return 0.5*cf[index] - (THETA/Ve_i[index])*(2+H[index])*(dVe_i[index])
                
                # define RK4 slope function for VeThetaH1
                def dVeThetaH1_by_dx(index, X, THETA, VETHETAH1):
                    return Ve_i[index]*0.0306*(((VETHETAH1/(Ve_i[index]*THETA))-3)**-0.6169)
                
                if case in wrong_columns:
                    continue 
                
                # Extract full column
                x_full   = TURBULENT_COORD[:, case, cpt]
                ve_full  = VE_I[:, case, cpt]
                dve_full = DVE_I[:, case, cpt]

                # Mask = NaNs
                mask = rp.isnan(x_full)

                # Extract valid entries
                x_i   = x_full[~mask]
                Ve_i  = ve_full[~mask]
                dVe_i = dve_full[~mask]

                Re_L         = RE_L[case,cpt]
                n            = len(x_i)
                dx           = rp.diff(x_i)
                nu           = NU[case,cpt]
                
                H            = rp.zeros(n) 
                H = H.at[0].set(ShapeFactor_0[case, cpt])
                Theta        = rp.zeros(n)
                Theta = Theta.at[0].set(THETA_0[case, cpt])
                H1           = rp.zeros(n) 
                H1 = H1.at[0].set((DEL_0[case, cpt] - DELTA_STAR_0[case, cpt]) / THETA_0[case, cpt])
                H1 = rp.where(H1[0] < 3.3, H1.at[0].set(3.417285), H1)
                
                cf           = rp.zeros(n)
                cf = cf.at[0].set(CF_0[case, cpt])
                VeThetaH1    = rp.zeros(n)
                VeThetaH1 = VeThetaH1.at[0].set(Ve_i[0] * Theta[0] * H1[0])
                
                for i in range(1,n):
                    # initialise the variable values at the current grid point using previous grid points (to define the error functions)
                    H_er = H[i-1];  cf_er = cf[i-1];  H1_er = H1[i-1];  Theta_er = Theta[i-1]
                    # assign previous grid point values of H and Cf to start RK4
                    H = H.at[i].set(H[i - 1])
                    
                    #assume some error values
                    erH = 0.2; erH1 = 0.2; erTheta = 0.2; ercf = 0.2;
                    
                    # iterate to get the variables at the grid point
                    while abs(erH)>0.00001 or abs(erH1)>0.00001 or abs(erTheta)>0.00001 or abs(ercf)>0.00001:
                        
                        # get Theta and VeThetaH1
                        Theta_i, VeThetaH1_i = RK4(i-1, dx, x_i, Theta, VeThetaH1, dTheta_by_dx, dVeThetaH1_by_dx)
                        Theta = Theta.at[i].set(Theta_i)
                        VeThetaH1 = VeThetaH1.at[i].set(VeThetaH1_i)
                        VeThetaH1 = rp.where(rp.isnan(VeThetaH1), rp.roll(VeThetaH1, 1), VeThetaH1)
                       
                        # get H1
                        H1 = H1.at[i].set(VeThetaH1[i] / (Ve_i[i] * Theta[i]))
                        
                        # get H
                        H = H.at[i].set(getH(H1[i]))
                        
                        # get skin friction
                        cf = cf.at[i].set(getcf(i, nu, H[i], Theta[i]))
                        
                        # define errors
                        erH = (H[i]-H_er)/H[i];
                        erH1 = (H1[i]-H1_er)/H1[i];
                        erTheta = (Theta[i]-Theta_er)/Theta[i];
                        ercf = (cf[i]-cf_er)/cf[i];
                        
                        # assign current iteration variable values to the Var_er
                        H_er = H[i]
                        H1_er = H1[i]
                        Theta_er = Theta[i]
                        cf_er = cf[i]
                
                
                delta_star   = H*Theta
                Re_theta     = Ve_i*Theta/nu
                Re_x         = (Ve_i*x_i)/nu
                delta        = (Theta*H1) + delta_star
                
                # Extract valid indices (non‑NaN)
                mask = rp.isnan(TURBULENT_COORD[:, case, cpt])
                indices = rp.nonzero(~mask)[0]

                # Scatter updates back into full arrays
                X_H          = X_H.at[indices, case, cpt].set(x_i)
                THETA_H      = THETA_H.at[indices, case, cpt].set(Theta)
                DELTA_STAR_H = DELTA_STAR_H.at[indices, case, cpt].set(delta_star)
                H_H          = H_H.at[indices, case, cpt].set(H)
                CF_H         = CF_H.at[indices, case, cpt].set(cf)
                RE_THETA_H   = RE_THETA_H.at[indices, case, cpt].set(Re_theta)
                RE_X_H       = RE_X_H.at[indices, case, cpt].set(Re_x)
                DELTA_H      = DELTA_H.at[indices, case, cpt].set(delta)


    RESULTS = Data(
            X_H          = X_H,      
            THETA_H      = THETA_H,   
            DELTA_STAR_H = DELTA_STAR_H,
            H_H          = H_H,       
            CF_H         = CF_H,   
            RE_THETA_H   = RE_THETA_H,   
            RE_X_H       = RE_X_H,    
            DELTA_H      = DELTA_H,   
        )    

    return  RESULTS


def RK4(ind, dx, x, Theta_var, VeThetaH1_var, Theta_slope, VeThetaH1_slope):
    k1 = Theta_slope(ind,  x[ind],  Theta_var[ind],  VeThetaH1_var[ind])
    l1 = VeThetaH1_slope(ind,  x[ind],  Theta_var[ind],  VeThetaH1_var[ind])
    
    k2 = Theta_slope(ind,  x[ind] + (dx[ind]/2),  Theta_var[ind] + (k1*dx[ind]/2),  VeThetaH1_var[ind] + (l1*dx[ind]/2))
    l2 = VeThetaH1_slope(ind,  x[ind] + (dx[ind]/2),  Theta_var[ind] + (k1*dx[ind]/2),  VeThetaH1_var[ind] + (l1*dx[ind]/2))
    
    k3 = Theta_slope(ind,  x[ind] + (dx[ind]/2),  Theta_var[ind] + (k2*dx[ind]/2),  VeThetaH1_var[ind] + (l2*dx[ind]/2))
    l3 = VeThetaH1_slope(ind,  x[ind] + (dx[ind]/2),  Theta_var[ind] + (k2*dx[ind]/2),  VeThetaH1_var[ind] + (l2*dx[ind]/2))
    
    k4 = Theta_slope(ind,  x[ind] + dx[ind],  Theta_var[ind] + (k3*dx[ind]),  VeThetaH1_var[ind] + (l2*dx[ind]))
    l4 = VeThetaH1_slope(ind,  x[ind] + dx[ind],  Theta_var[ind] + (k3*dx[ind]),  VeThetaH1_var[ind] + (l2*dx[ind]))
    
    Theta_new = Theta_var[ind] + ((dx[ind]/6)*(k1 + 2*k2 + 2*k3 + k4))
    VeThetaH1_new = VeThetaH1_var[ind] + ((dx[ind]/6)*(l1 + 2*l2 + 2*l3 + l4))
    return Theta_new, VeThetaH1_new 