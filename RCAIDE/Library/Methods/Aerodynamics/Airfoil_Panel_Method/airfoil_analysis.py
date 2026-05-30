# RCAIDE/Methods/Aerodynamics/Airfoil_Panel_Method/airfoil_analysis.py
# 
# 
# Created:  Dec 2023, M. Clarke
# Modified: Apr 2024, N. Nanjappa

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
# RCAIDE imports  
from RCAIDE.Framework.Core import Data
from .hess_smith           import hess_smith
from .thwaites_method      import thwaites_method 
from .heads_method         import heads_method 
from .aero_coeff           import aero_coeff
from .chordwise_distribution import chordwise_distribution
from .cf_filter            import cf_filter

# pacakge imports  
import RNUMPY as rp  

# ----------------------------------------------------------------------------------------------------------------------
# airfoil_analysis.py
# ----------------------------------------------------------------------------------------------------------------------

def airfoil_analysis(airfoil_geometry, alpha, Re_L,
                     initial_momentum_thickness=1e-5,
                     tolerance=1e0,
                     H_wake=1.05,
                     Ue_wake=0.99):
    """This computes the aerodynamic polars as well as the boundary layer properties of 
    an airfoil at a defined set of reynolds numbers and angle of attacks

    Assumptions:
    Michel Criteria used for transition 
    
    Squire-Young relation for total drag (exrapolates theta from end of wake). 
    However, since we do not have a wake we will assume H_wake = 1.05 and Ue_wake = 0.99
    The freestream velocity is taken to be 1 m/s.
    Airfoil has a unit chord length [1 m].

    Source:
    N/A

    Inputs: 
    airfoil_geometry   - airfoil geometry points                                                             [unitless]
    alpha              - angle of attacks                                                                    [radians]
    Re_L               - Reynolds numbers                                                                    [unitless]
    batch_analysis     - boolean : If True: the specified number of angle of attacks and Reynolds            [boolean]
                                  numbers are used to create a table of 2-D results for each combination
                                  Note: Can only accomodate one airfoil
                                  
                                  If False:The airfoils specified are run and corresponding angle of attacks 
                                  and Reynolds numbers
                                  Note: The number of airfoils, angle of attacks and reynolds numbers must 
                                  all the same dimension                     
    
    Outputs: 
    airfoil_properties.
        AoA            - angle of attack                                                   [radians
        Re             - Reynolds number                                                   [unitless]
        Cl             - lift coefficients                                                 [unitless]
        Cd             - drag coefficients                                                 [unitless]
        Cm             - moment coefficients                                               [unitless]
        normals        - surface normals of airfoil                                        [unitless]
        x              - x coordinate points on airfoil                                    [unitless]
        y              - y coordinate points on airfoil                                    [unitless]
        x_bl           - x coordinate points on airfoil adjusted to include boundary layer [unitless]
        y_bl           - y coordinate points on airfoil adjusted to include boundary layer [unitless]
        Cp             - pressure coefficient distribution                                 [unitless]
        Ue_Vinf        - ratio of boundary layer edge velocity to freestream               [unitless]
        dVe            - derivative of boundary layer velocity                             [m/s-m]
        theta          - momentum thickness                                                [m]
        delta_star     - displacement thickness                                            [m]
        delta          - boundary layer thickness                                          [m]
        H              - shape factor                                                      [unitless]
        Cf             - local skin friction coefficient                                   [unitless]
        Re_theta_t     - Reynolds Number as a function of theta transition location        [unitless]
        tr_crit        - critical transition criteria                                      [unitless]
                        
    Properties Used:
    N/A
    """   

    ncases  = len(alpha[0, :])
    ncpts   = len(Re_L)
    x_coord = airfoil_geometry.x_coordinates
    y_coord = airfoil_geometry.y_coordinates
    npanel  = len(x_coord) - 1

    x_coord_3d = rp.tile(x_coord[:, None, None], (1, ncases, ncpts))
    y_coord_3d = rp.tile(y_coord[:, None, None], (1, ncases, ncpts))

    X, Y, vt, normals = hess_smith(x_coord_3d, y_coord_3d, alpha, Re_L, npanel)

    RE_L_VALS = Re_L.T
    nu        = 1.0 / RE_L_VALS

    # ---------------------------------------------------------------------
    # Bottom surface (vt < 0)
    # ---------------------------------------------------------------------
    VT_mask_bot = vt > 0.0
    VT_bot      = rp.where(VT_mask_bot, rp.nan, vt)
    X_BOT_VALS  = rp.where(VT_mask_bot, rp.nan, X)[::-1]
    Y_BOT       = rp.where(VT_mask_bot, rp.nan, Y)[::-1]

    X_BOT = rp.zeros_like(X_BOT_VALS)
    dX    = X_BOT_VALS[1:] - X_BOT_VALS[:-1]
    dY    = Y_BOT[1:]      - Y_BOT[:-1]
    ds    = rp.sqrt(dX**2 + dY**2)
    # nan_to_num prevents NaN propagation through the top-surface gap in ds
    X_BOT = X_BOT.at[1:].set(rp.cumsum(rp.nan_to_num(ds), axis=0))
    # Re-apply the original mask: top-surface panels must stay NaN in X_BOT
    X_BOT = rp.where(rp.isnan(VT_bot[::-1]), rp.nan, X_BOT)

    X_BOT_mask = rp.isnan(X_BOT)
    VT_bot_mask = rp.isnan(VT_bot[::-1])

    first_idx  = rp.sum(X_BOT_mask, axis=0)
    mask_count = X_BOT.shape[0] - first_idx
    prev_index = first_idx - 1

    first_panel    = prev_index.flatten()
    last_panel     = (first_idx - 1 + mask_count).flatten()
    last_paneldve  = (first_idx - 2 + mask_count).flatten()
    aoas           = rp.repeat(rp.arange(ncases), ncpts)
    res            = rp.tile(rp.arange(ncpts), ncases)

    # Unmask first panel in X_BOT
    X_BOT_mask = X_BOT_mask.at[first_panel, aoas, res].set(False)
    X_BOT      = rp.where(X_BOT_mask, rp.nan, X_BOT)

    VE_BOT = rp.abs(vt[::-1])
    # Use the actual vt value from Hess-Smith at the first_panel position 
    # (the LE), rather than copying from the next panel. (Note: first_panel 
    # refers to the stagnation point in the surface-specific coordinates)
    VE_BOT = VE_BOT.at[first_panel, aoas, res].set(
                 rp.abs(vt[::-1][first_panel, aoas, res]))

    # DVE_BOT
    DVE_BOT      = rp.zeros_like(X_BOT)
    dVE          = rp.diff(VE_BOT, axis=0)
    dX_BOT       = rp.diff(X_BOT, axis=0)
    DVE_TEMP     = dVE / dX_BOT

    a = X_BOT[1:-1] - X_BOT[:-2]
    b = X_BOT[2:]   - X_BOT[:-2]
    DVE_mid = (b * DVE_TEMP[:-1] + a * DVE_TEMP[1:]) / (a + b)
    DVE_BOT = DVE_BOT.at[1:-1].set(DVE_mid)

    DVE_BOT_mask = rp.isnan(X_BOT)
    DVE_BOT = rp.where(DVE_BOT_mask, rp.nan, DVE_BOT)
    # first_panel endpoint: use DVE_TEMP[first_panel] which is now finite
    DVE_BOT = DVE_BOT.at[first_panel, aoas, res].set(DVE_TEMP[first_panel, aoas, res])
    # last_panel endpoint: clamp index into DVE_TEMP (shape npanel-1)
    safe_last_dve = rp.clip(last_paneldve, 0, DVE_TEMP.shape[0] - 1)
    DVE_BOT = DVE_BOT.at[last_panel, aoas, res].set(DVE_TEMP[safe_last_dve, aoas, res])

    L_BOT = rp.nanmax(X_BOT, axis=0)

    # High-AOA wrong columns
    # AOA_deg       = rp.rad2deg(alpha)
    # wrong_columns = rp.where(rp.abs(AOA_deg) > 80.0)[1]
    AOA_deg = rp.rad2deg(alpha)
    wrong_columns = rp.nonzero(rp.abs(AOA_deg) > 80.0)[1]

    # X_BOT already has its valid region in LE→TE order:
    #   first_panel (≈ npanel/2) = arc-length 0 (bottom LE)
    #   npanel-1                 = arc-length L (bottom TE)
    # No additional flip is needed — matches how the top surface is handled.
    BOT_T_RESULTS = thwaites_method(
        npanel, ncases, ncpts, nu, L_BOT, RE_L_VALS,
        X_BOT, VE_BOT, DVE_BOT, tolerance, wrong_columns,
        THETA_0=initial_momentum_thickness
    )

    # Results are in reversed-HS order (valid at first_panel..npanel-1) — no flip needed
    X_T_BOT          = BOT_T_RESULTS.X_T
    THETA_T_BOT      = BOT_T_RESULTS.THETA_T
    DELTA_STAR_T_BOT = BOT_T_RESULTS.DELTA_STAR_T
    H_T_BOT          = BOT_T_RESULTS.H_T
    CF_T_BOT         = BOT_T_RESULTS.CF_T
    RE_THETA_T_BOT   = BOT_T_RESULTS.RE_THETA_T
    RE_X_T_BOT       = BOT_T_RESULTS.RE_X_T
    DELTA_T_BOT      = BOT_T_RESULTS.DELTA_T

    TR_CRIT_BOT    = RE_THETA_T_BOT - 1.174 * (1.0 + 22400.0 / RE_X_T_BOT) * RE_X_T_BOT**0.46
    CRIT_mask_BOT  = TR_CRIT_BOT > 0.0
    # transition_panel must be the absolute index within the npanel array.
    # We count leading NaNs (ghost panels from other surface) and then add 
    # the count of valid laminar panels.
    n_leading_nans   = rp.sum(rp.isnan(X_BOT), axis=0)
    n_laminar        = rp.sum(~CRIT_mask_BOT & ~rp.isnan(X_BOT), axis=0)
    
    # Delay transition by one panel: the transition panel is now the first 
    # turbulent node instead of the last laminar node. This reduces the total
    # turbulent growth length slightly to match verification data.
    transition_panel = rp.clip(n_leading_nans + n_laminar, n_leading_nans, rp.ones_like(n_leading_nans)*(npanel - 1))
    transition_panel = transition_panel.flatten()
    aoas             = rp.repeat(rp.arange(ncases), ncpts)
    res              = rp.tile(rp.arange(ncpts), ncases)

    X_TR_BOT          = X_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)
    DELTA_STAR_TR_BOT = DELTA_STAR_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)
    THETA_TR_BOT      = THETA_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)
    DELTA_TR_BOT      = DELTA_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)
    CF_TR_BOT         = CF_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)
    H_TR_BOT          = H_T_BOT[transition_panel, aoas, res].reshape(ncases, ncpts)

    TURBULENT_SURF  = L_BOT
    TURBULENT_COORD = X_BOT - X_TR_BOT
    TURBULENT_COORD = rp.where(TURBULENT_COORD < 0.0, rp.nan, TURBULENT_COORD)

    # Pass TURBULENT_COORD and VE_BOT/DVE_BOT directly (already LE→TE in valid region)
    BOT_H_RESULTS = heads_method(
        npanel, ncases, ncpts, nu,
        DELTA_TR_BOT, THETA_TR_BOT, DELTA_STAR_TR_BOT,
        CF_TR_BOT, H_TR_BOT, RE_L_VALS,
        TURBULENT_COORD,
        VE_BOT, DVE_BOT, TURBULENT_SURF,
        tolerance, wrong_columns
    )

    # Results are in reversed-HS order — no flip needed
    X_H_BOT          = BOT_H_RESULTS.X_H
    THETA_H_BOT      = BOT_H_RESULTS.THETA_H
    DELTA_STAR_H_BOT = BOT_H_RESULTS.DELTA_STAR_H
    H_H_BOT          = BOT_H_RESULTS.H_H
    CF_H_BOT         = BOT_H_RESULTS.CF_H
    RE_THETA_H_BOT   = BOT_H_RESULTS.RE_THETA_H
    RE_X_H_BOT       = BOT_H_RESULTS.RE_X_H
    DELTA_H_BOT      = BOT_H_RESULTS.DELTA_H

    # Apply masks via NaNs
    X_T_BOT          = rp.where(CRIT_mask_BOT, rp.nan, X_T_BOT)
    THETA_T_BOT      = rp.where(CRIT_mask_BOT, rp.nan, THETA_T_BOT)
    DELTA_STAR_T_BOT = rp.where(CRIT_mask_BOT, rp.nan, DELTA_STAR_T_BOT)
    H_T_BOT          = rp.where(CRIT_mask_BOT, rp.nan, H_T_BOT)
    CF_T_BOT         = rp.where(CRIT_mask_BOT, rp.nan, CF_T_BOT)
    RE_THETA_T_BOT   = rp.where(CRIT_mask_BOT, rp.nan, RE_THETA_T_BOT)
    RE_X_T_BOT       = rp.where(CRIT_mask_BOT, rp.nan, RE_X_T_BOT)
    DELTA_T_BOT      = rp.where(CRIT_mask_BOT, rp.nan, DELTA_T_BOT)

    inv_CRIT_mask_BOT = ~CRIT_mask_BOT

    # Keep heads (turbulent) results only where CRIT_mask_BOT is True (turbulent).
    # Previously these were masked with inv_CRIT_mask_BOT which was backwards.
    X_H_BOT          = rp.where(CRIT_mask_BOT, X_H_BOT, rp.nan)
    THETA_H_BOT      = rp.where(CRIT_mask_BOT, THETA_H_BOT, rp.nan)
    DELTA_STAR_H_BOT = rp.where(CRIT_mask_BOT, DELTA_STAR_H_BOT, rp.nan)
    H_H_BOT          = rp.where(CRIT_mask_BOT, H_H_BOT, rp.nan)
    CF_H_BOT         = rp.where(CRIT_mask_BOT, CF_H_BOT, rp.nan)
    RE_THETA_H_BOT   = rp.where(CRIT_mask_BOT, RE_THETA_H_BOT, rp.nan)
    RE_X_H_BOT       = rp.where(CRIT_mask_BOT, RE_X_H_BOT, rp.nan)
    DELTA_H_BOT      = rp.where(CRIT_mask_BOT, DELTA_H_BOT, rp.nan)

    # laminar before transition, turbulent after
    X_BOT_SURF          = rp.where(CRIT_mask_BOT, X_H_BOT, X_T_BOT)
    THETA_BOT_SURF      = rp.where(CRIT_mask_BOT, THETA_H_BOT, THETA_T_BOT)
    DELTA_STAR_BOT_SURF = rp.where(CRIT_mask_BOT, DELTA_STAR_H_BOT, DELTA_STAR_T_BOT)
    H_BOT_SURF          = rp.where(CRIT_mask_BOT, H_H_BOT, H_T_BOT)
    CF_BOT_SURF         = rp.where(CRIT_mask_BOT, CF_H_BOT, CF_T_BOT)
    RE_THETA_BOT_SURF   = rp.where(CRIT_mask_BOT, RE_THETA_H_BOT, RE_THETA_T_BOT)
    RE_X_BOT_SURF       = rp.where(CRIT_mask_BOT, RE_X_H_BOT, RE_X_T_BOT)
    DELTA_BOT_SURF      = rp.where(CRIT_mask_BOT, DELTA_H_BOT, DELTA_T_BOT)



    # ---------------------------------------------------------------------
    # Top surface (vt > 0)
    # ---------------------------------------------------------------------
    VT_mask_top = vt < 0.0
    VT_top      = rp.where(VT_mask_top, rp.nan, vt)
    CP          = 1.0 - vt**2

    X_TOP_VALS = rp.where(VT_mask_top, rp.nan, X)
    Y_TOP      = rp.where(VT_mask_top, rp.nan, Y)

    X_TOP = rp.zeros_like(X_TOP_VALS)
    dX    = X_TOP_VALS[1:] - X_TOP_VALS[:-1]
    dY    = Y_TOP[1:]      - Y_TOP[:-1]
    ds    = rp.sqrt(dX**2 + dY**2)
    X_TOP = X_TOP.at[1:].set(rp.cumsum(rp.nan_to_num(ds), axis=0))
    # Re-apply the original mask: bottom-surface panels must stay NaN in X_TOP
    X_TOP = rp.where(rp.isnan(VT_top), rp.nan, X_TOP)

    X_TOP_mask = rp.isnan(X_TOP)
    first_idx  = rp.sum(X_TOP_mask, axis=0)
    mask_count = X_TOP.shape[0] - first_idx
    prev_index = first_idx - 1

    first_panel    = prev_index.flatten()
    last_panel     = (first_idx - 1 + mask_count).flatten()
    last_paneldve  = (first_idx - 2 + mask_count).flatten()
    aoas           = rp.repeat(rp.arange(ncases), ncpts)
    res            = rp.tile(rp.arange(ncpts), ncases)

    X_TOP_mask = X_TOP_mask.at[first_panel, aoas, res].set(False)
    X_TOP      = rp.where(X_TOP_mask, rp.nan, X_TOP)

    VE_TOP = rp.abs(vt)
    # Use the actual vt value from Hess-Smith at the first_panel position 
    # (the LE), rather than copying from the next panel.
    VE_TOP = VE_TOP.at[first_panel, aoas, res].set(
                 rp.abs(vt[first_panel, aoas, res]))

    DVE_TOP      = rp.zeros_like(X_TOP)
    dVE          = rp.diff(VE_TOP, axis=0)
    dX_TOP       = rp.diff(X_TOP, axis=0)
    DVE_TEMP     = dVE / dX_TOP

    a = X_TOP[1:-1] - X_TOP[:-2]
    b = X_TOP[2:]   - X_TOP[:-2]
    DVE_mid = (b * DVE_TEMP[:-1] + a * DVE_TEMP[1:]) / (a + b)
    DVE_TOP = DVE_TOP.at[1:-1].set(DVE_mid)

    DVE_TOP_mask = rp.isnan(X_TOP)
    DVE_TOP = rp.where(DVE_TOP_mask, rp.nan, DVE_TOP)
    DVE_TOP = DVE_TOP.at[first_panel, aoas, res].set(DVE_TEMP[first_panel, aoas, res])
    # last_panel endpoint: clamp index into DVE_TEMP (shape npanel-1)
    safe_last_dve = rp.clip(last_paneldve, 0, DVE_TEMP.shape[0] - 1)
    DVE_TOP = DVE_TOP.at[last_panel, aoas, res].set(DVE_TEMP[safe_last_dve, aoas, res])

    L_TOP = rp.nanmax(X_TOP, axis=0)

    TOP_T_RESULTS = thwaites_method(
        npanel, ncases, ncpts, nu, L_TOP, RE_L_VALS,
        X_TOP, VE_TOP, DVE_TOP, tolerance, wrong_columns,
        THETA_0=initial_momentum_thickness
    )

    X_T_TOP          = TOP_T_RESULTS.X_T
    THETA_T_TOP      = TOP_T_RESULTS.THETA_T
    DELTA_STAR_T_TOP = TOP_T_RESULTS.DELTA_STAR_T
    H_T_TOP          = TOP_T_RESULTS.H_T
    CF_T_TOP         = TOP_T_RESULTS.CF_T
    RE_THETA_T_TOP   = TOP_T_RESULTS.RE_THETA_T
    RE_X_T_TOP       = TOP_T_RESULTS.RE_X_T
    DELTA_T_TOP      = TOP_T_RESULTS.DELTA_T

    TR_CRIT_TOP   = RE_THETA_T_TOP - 1.174 * (1.0 + 22400.0 / RE_X_T_TOP) * (RE_X_T_TOP**0.46)
    CRIT_mask_TOP = TR_CRIT_TOP > 0.0

    # Shift transition point forward by one panel
    n_leading_nans   = rp.sum(rp.isnan(X_TOP), axis=0)
    n_laminar        = rp.sum(~CRIT_mask_TOP & ~rp.isnan(X_TOP), axis=0)
    
    transition_panel = rp.clip(n_leading_nans + n_laminar, n_leading_nans, rp.ones_like(n_leading_nans)*(npanel - 1))
    transition_panel = transition_panel.flatten()
    aoas             = rp.repeat(rp.arange(ncases), ncpts)
    res              = rp.tile(rp.arange(ncpts), ncases)

    X_TR_TOP          = X_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)
    DELTA_STAR_TR_TOP = DELTA_STAR_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)
    THETA_TR_TOP      = THETA_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)
    DELTA_TR_TOP      = DELTA_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)
    CF_TR_TOP         = CF_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)
    H_TR_TOP          = H_T_TOP[transition_panel, aoas, res].reshape(ncases, ncpts)

    TURBULENT_SURF  = L_TOP
    TURBULENT_COORD = X_TOP - X_TR_TOP
    TURBULENT_COORD = rp.where(TURBULENT_COORD < 0.0, rp.nan, TURBULENT_COORD)

    TOP_H_RESULTS = heads_method(
        npanel, ncases, ncpts, nu,
        DELTA_TR_TOP, THETA_TR_TOP, DELTA_STAR_TR_TOP,
        CF_TR_TOP, H_TR_TOP, RE_L_VALS,
        TURBULENT_COORD, VE_TOP, DVE_TOP, TURBULENT_SURF,
        tolerance, wrong_columns
    )

    X_H_TOP          = TOP_H_RESULTS.X_H
    THETA_H_TOP      = TOP_H_RESULTS.THETA_H
    DELTA_STAR_H_TOP = TOP_H_RESULTS.DELTA_STAR_H
    H_H_TOP          = TOP_H_RESULTS.H_H
    CF_H_TOP         = TOP_H_RESULTS.CF_H
    RE_THETA_H_TOP   = TOP_H_RESULTS.RE_THETA_H
    RE_X_H_TOP       = TOP_H_RESULTS.RE_X_H
    DELTA_H_TOP      = TOP_H_RESULTS.DELTA_H

    # Build final top surface directly: laminar before transition, turbulent after
    X_H_TOP_MOD = X_H_TOP + X_TR_TOP

    X_TOP_SURF          = rp.where(CRIT_mask_TOP, X_H_TOP_MOD, X_T_TOP)
    THETA_TOP_SURF      = rp.where(CRIT_mask_TOP, THETA_H_TOP, THETA_T_TOP)
    DELTA_STAR_TOP_SURF = rp.where(CRIT_mask_TOP, DELTA_STAR_H_TOP, DELTA_STAR_T_TOP)
    H_TOP_SURF          = rp.where(CRIT_mask_TOP, H_H_TOP, H_T_TOP)
    CF_TOP_SURF         = rp.where(CRIT_mask_TOP, CF_H_TOP, CF_T_TOP)
    RE_THETA_TOP_SURF   = rp.where(CRIT_mask_TOP, RE_THETA_H_TOP, RE_THETA_T_TOP)
    RE_X_TOP_SURF       = rp.where(CRIT_mask_TOP, RE_X_H_TOP, RE_X_T_TOP)
    DELTA_TOP_SURF      = rp.where(CRIT_mask_TOP, DELTA_H_TOP, DELTA_T_TOP)

    # ----------------------------------------------------------------------
    # Concatenate lower and upper surfaces
    # ----------------------------------------------------------------------
    # To correctly merge into a single Hess-Smith array, the bottom surface 
    # properties (currently reversed LE->TE) must be flipped back. 
    # This puts them back into their original index range (roughly 0..first_panel).
    # vt < 0 → bottom surface, vt >= 0 → top surface.
    THETA      = rp.where(vt < 0.0, rp.flip(THETA_BOT_SURF, axis=0),      THETA_TOP_SURF)
    DELTA_STAR = rp.where(vt < 0.0, rp.flip(DELTA_STAR_BOT_SURF, axis=0), DELTA_STAR_TOP_SURF)
    H          = rp.where(vt < 0.0, rp.flip(H_BOT_SURF, axis=0),          H_TOP_SURF)
    CF         = rp.where(vt < 0.0, rp.flip(CF_BOT_SURF, axis=0),         CF_TOP_SURF)
    RE_THETA   = rp.where(vt < 0.0, rp.flip(RE_THETA_BOT_SURF, axis=0),   RE_THETA_TOP_SURF)
    RE_X       = rp.where(vt < 0.0, rp.flip(RE_X_BOT_SURF, axis=0),       RE_X_TOP_SURF)
    DELTA      = rp.where(vt < 0.0, rp.flip(DELTA_BOT_SURF, axis=0),      DELTA_TOP_SURF)

    # ----------------------------------------------------------------------
    # Build VE and DVE on the full surface (bottom + top) in original HS order
    # ----------------------------------------------------------------------
    # IMPORTANT: X_BOT is in *reversed* HS order (built with [::-1]), so
    # concatenate_surfaces' bot_mask would index VE_BOT_flipped at the wrong
    # physical panel.  Instead, compute VE/DVE directly in original HS order:
    #
    #   VE[i]  = |vt[i]|  — tangential speed, already HS-panel indexed, no NaN
    #   DVE[i] = dVE/ds at HS panel i, from the per-surface derivative arrays
    #             (flip DVE_BOT from reversed→HS order; DVE_TOP is already HS order)

    VE  = rp.abs(vt)    # (npanel, ncases, ncpts), non-NaN everywhere

    # vt < 0 → bottom panel in original HS order → use flipped DVE_BOT
    # vt >= 0 → top panel in original HS order   → use DVE_TOP (already HS order)
    DVE = rp.where(vt < 0,
                   rp.flip(DVE_BOT, axis=0),
                   DVE_TOP)

    # Filter skin friction
    CF = cf_filter(ncpts, ncases, npanel, CF)

    # ----------------------------------------------------------------------
    # Recompute geometry with boundary layer displacement thickness
    # ----------------------------------------------------------------------
    DELTA         = rp.nan_to_num(DELTA)
    y_coord_3d_bl = Y + DELTA * normals[:, 1, :, :]
    x_coord_3d_bl = X + DELTA * normals[:, 0, :, :]
    npanel_mod    = npanel - 1

    X_BL, Y_BL, vt_bl, normals_bl = hess_smith(
        x_coord_3d_bl, y_coord_3d_bl, alpha, Re_L, npanel_mod
    )

    # ----------------------------------------------------------------------
    # Bottom BL
    # ----------------------------------------------------------------------
    VT_BL_mask_bot = vt_bl > 0.0
    VT_BL_bot      = rp.where(VT_BL_mask_bot, rp.nan, vt_bl)
    X_BL_BOT_VALS  = rp.where(VT_BL_mask_bot, rp.nan, X_BL)[::-1]
    Y_BL_BOT       = rp.where(VT_BL_mask_bot, rp.nan, Y_BL)[::-1]

    X_BL_BOT = rp.zeros_like(X_BL_BOT_VALS)
    dX       = X_BL_BOT_VALS[1:] - X_BL_BOT_VALS[:-1]
    dY       = Y_BL_BOT[1:]      - Y_BL_BOT[:-1]
    ds       = rp.sqrt(dX**2 + dY**2)
    X_BL_BOT = X_BL_BOT.at[1:].set(rp.cumsum(rp.nan_to_num(ds), axis=0))
    X_BL_BOT = rp.where(rp.isnan(VT_BL_bot[::-1]), rp.nan, X_BL_BOT)

    X_BL_BOT_mask = rp.isnan(X_BL_BOT)
    first_idx     = rp.sum(X_BL_BOT_mask, axis=0)
    mask_count    = X_BL_BOT.shape[0] - first_idx
    prev_index    = first_idx - 1

    first_panel    = prev_index.flatten()
    last_panel     = (first_idx - 1 + mask_count).flatten()
    last_paneldve  = (first_idx - 2 + mask_count).flatten()
    aoas           = rp.repeat(rp.arange(ncases), ncpts)
    res            = rp.tile(rp.arange(ncpts), ncases)

    X_BL_BOT_mask = X_BL_BOT_mask.at[first_panel, aoas, res].set(False)
    X_BL_BOT      = rp.where(X_BL_BOT_mask, rp.nan, X_BL_BOT)

    VE_BL_BOT = -VT_BL_bot[::-1]
    CP_BL_BOT = 1.0 - VE_BL_BOT**2

    # ----------------------------------------------------------------------
    # Top BL
    # ----------------------------------------------------------------------
    VT_BL_mask_top = vt_bl < 0.0
    VT_BL_top      = rp.where(VT_BL_mask_top, rp.nan, vt_bl)
    X_BL_TOP_VALS  = rp.where(VT_BL_mask_top, rp.nan, X_BL)
    Y_BL_TOP       = rp.where(VT_BL_mask_top, rp.nan, Y_BL)

    X_BL_TOP = rp.zeros_like(X_BL_TOP_VALS)
    dX       = X_BL_TOP_VALS[1:] - X_BL_TOP_VALS[:-1]
    dY       = Y_BL_TOP[1:]      - Y_BL_TOP[:-1]
    ds       = rp.sqrt(dX**2 + dY**2)
    X_BL_TOP = X_BL_TOP.at[1:].set(rp.cumsum(rp.nan_to_num(ds), axis=0))
    X_BL_TOP = rp.where(rp.isnan(VT_BL_top), rp.nan, X_BL_TOP)

    X_BL_TOP_mask = rp.isnan(X_BL_TOP)
    first_idx     = rp.sum(X_BL_TOP_mask, axis=0)
    mask_count    = X_BL_TOP.shape[0] - first_idx
    prev_index    = first_idx - 1

    first_panel    = prev_index.flatten()
    last_panel     = (first_idx - 1 + mask_count).flatten()
    last_paneldve  = (first_idx - 2 + mask_count).flatten()
    aoas           = rp.repeat(rp.arange(ncases), ncpts)
    res            = rp.tile(rp.arange(ncpts), ncases)

    X_BL_TOP_mask = X_BL_TOP_mask.at[first_panel, aoas, res].set(False)
    X_BL_TOP      = rp.where(X_BL_TOP_mask, rp.nan, X_BL_TOP)

    VE_BL_TOP = VT_BL_top
    CP_BL_TOP = 1.0 - VE_BL_TOP**2

    # ----------------------------------------------------------------------
    # Build CP_BL and X_BL on full BL surface (bottom + top)
    # ----------------------------------------------------------------------
    CP_BL = concatenate_surfaces(
        X_BL_BOT, X_BL_TOP,
        rp.flip(CP_BL_BOT, axis=0), CP_BL_TOP,
        npanel_mod, ncases, ncpts, wrong_columns
    )

    X_BL_SURF = concatenate_surfaces(
        X_BL_BOT, X_BL_TOP,
        rp.flip(X_BL_BOT, axis=0), X_BL_TOP,
        npanel_mod, ncases, ncpts, wrong_columns
    )

    DCP_DX = rp.diff(CP_BL, axis=0) / rp.diff(X_BL_SURF, axis=0)

    # ----------------------------------------------------------------------
    # Inviscid coefficients and Squire–Young drag
    # ----------------------------------------------------------------------
    AERO_RES = aero_coeff(x_coord_3d, y_coord_3d, CP, alpha, npanel)

    # Squire-Young uses trailing-edge theta, H, and VE on each surface.
    # THETA / H from concatenate_surfaces are in a mixed (reversed-bot /
    # forward-top) ordering that does NOT align with VE = abs(vt).  Read
    # the TE values directly from the per-surface arrays in their natural
    # storage order instead:
    #   Bottom surface: stored in *reversed* HS order → TE is the LAST row
    #   Top surface:    stored in *forward* HS order  → TE is the LAST row
    THETA_bot_TE = THETA_BOT_SURF[-1, :, :]   # (ncases, ncpts)
    H_bot_TE     = H_BOT_SURF[-1, :, :]
    VE_bot_TE    = VE_BOT[-1, :, :]           # reversed last row = bottom TE

    THETA_top_TE = THETA_TOP_SURF[-1, :, :]   # (ncases, ncpts)
    H_top_TE     = H_TOP_SURF[-1, :, :]
    VE_top_TE    = VE_TOP[-1, :, :]           # forward last row = top TE

    del2_inf_l = THETA_bot_TE * VE_bot_TE**((5.0 + H_bot_TE) / 2.0)
    del2_inf_u = THETA_top_TE * VE_top_TE**((5.0 + H_top_TE) / 2.0)
    del2_inf   = del2_inf_u + del2_inf_l
    cd_sqy     = 2.0 * del2_inf.T

    fL, fD = chordwise_distribution(
        x_coord_3d, y_coord_3d, CP, alpha, npanel, CF, vt
    )

    # ----------------------------------------------------------------------
    # Zero out wrong columns
    # ----------------------------------------------------------------------
    AERO_RES.cl   = AERO_RES.cl.at[:, wrong_columns].set(0.0)
    AERO_RES.cdpi = AERO_RES.cdpi.at[:, wrong_columns].set(0.0)
    AERO_RES.cm   = AERO_RES.cm.at[:, wrong_columns].set(0.0)
    cd_sqy        = cd_sqy.at[:, wrong_columns].set(0.0)
    CP_BL         = CP_BL.at[:, wrong_columns, :].set(0.0)
    DCP_DX        = DCP_DX.at[:, wrong_columns, :].set(0.0)
    VE            = VE.at[:, wrong_columns, :].set(0.0)
    DVE           = DVE.at[:, wrong_columns, :].set(0.0)
    THETA         = THETA.at[:, wrong_columns, :].set(0.0)
    DELTA_STAR    = DELTA_STAR.at[:, wrong_columns, :].set(0.0)
    DELTA         = DELTA.at[:, wrong_columns, :].set(0.0)
    RE_THETA      = RE_THETA.at[:, wrong_columns, :].set(0.0)
    RE_X          = RE_X.at[:, wrong_columns, :].set(0.0)
    H             = H.at[:, wrong_columns, :].set(0.0)
    CF            = CF.at[:, wrong_columns, :].set(0.0)
    fL            = fL.at[:, wrong_columns, :].set(0.0)
    fD            = fD.at[:, wrong_columns, :].set(0.0)

    #     flat = arr.reshape((-1,), order='F')
    #     flat = flat[~rp.isnan(flat)]
    #     return flat.reshape((npanel, ncases, ncpts), order='F')

    # VE  = drop_nans_and_reshape_full(VE_VALS)
    # DVE = drop_nans_and_reshape_full(DVE_VALS)

    # CF = cf_filter(ncpts, ncases, npanel, CF)

    # DELTA         = rp.nan_to_num(DELTA)
    # y_coord_3d_bl = Y + DELTA * normals[:, 1, :, :]
    # x_coord_3d_bl = X + DELTA * normals[:, 0, :, :]
    # npanel_mod    = npanel - 1

    # X_BL, Y_BL, vt_bl, normals_bl = hess_smith(x_coord_3d_bl, y_coord_3d_bl, alpha, Re_L, npanel_mod)

    # # Bottom BL
    # VT_BL_mask_bot = vt_bl > 0.0
    # VT_BL_bot      = rp.where(VT_BL_mask_bot, rp.nan, vt_bl)
    # X_BL_BOT_VALS  = rp.where(VT_BL_mask_bot, rp.nan, X_BL)[::-1]
    # Y_BL_BOT       = rp.where(VT_BL_mask_bot, rp.nan, Y_BL)[::-1]

    # X_BL_BOT = rp.zeros_like(X_BL_BOT_VALS)
    # dX       = X_BL_BOT_VALS[1:] - X_BL_BOT_VALS[:-1]
    # dY       = Y_BL_BOT[1:]      - Y_BL_BOT[:-1]
    # ds       = rp.sqrt(dX**2 + dY**2)
    # X_BL_BOT = X_BL_BOT.at[1:].set(rp.cumsum(ds, axis=0))

    # X_BL_BOT_mask = rp.isnan(X_BL_BOT)
    # first_idx     = rp.sum(X_BL_BOT_mask, axis=0)
    # mask_count    = X_BL_BOT.shape[0] - first_idx
    # prev_index    = first_idx - 1

    # first_panel    = list(prev_index.flatten())
    # last_panel     = list((first_idx - 1 + mask_count).flatten())
    # last_paneldve  = list((first_idx - 2 + mask_count).flatten())
    # aoas           = list(rp.repeat(rp.arange(ncases), ncpts))
    # res            = list(rp.tile(rp.arange(ncpts), ncases))

    # X_BL_BOT_mask = X_BL_BOT_mask.at[first_panel, aoas, res].set(False)
    # X_BL_BOT      = rp.where(X_BL_BOT_mask, rp.nan, X_BL_BOT)

    # VE_BL_BOT = -VT_BL_bot[::-1]
    # CP_BL_BOT = 1.0 - VE_BL_BOT**2

    # # Top BL
    # VT_BL_mask_top = vt_bl < 0.0
    # VT_BL_top      = rp.where(VT_BL_mask_top, rp.nan, vt_bl)
    # X_BL_TOP_VALS  = rp.where(VT_BL_mask_top, rp.nan, X_BL)
    # Y_BL_TOP       = rp.where(VT_BL_mask_top, rp.nan, Y_BL)

    # X_BL_TOP = rp.zeros_like(X_BL_TOP_VALS)
    # dX       = X_BL_TOP_VALS[1:] - X_BL_TOP_VALS[:-1]
    # dY       = Y_BL_TOP[1:]      - Y_BL_TOP[:-1]
    # ds       = rp.sqrt(dX**2 + dY**2)
    # X_BL_TOP = X_BL_TOP.at[1:].set(rp.cumsum(ds, axis=0))

    # X_BL_TOP_mask = rp.isnan(X_BL_TOP)
    # first_idx     = rp.sum(X_BL_TOP_mask, axis=0)
    # mask_count    = X_BL_TOP.shape[0] - first_idx
    # prev_index    = first_idx - 1

    # first_panel    = list(prev_index.flatten())
    # last_panel     = list((first_idx - 1 + mask_count).flatten())
    # last_paneldve  = list((first_idx - 2 + mask_count).flatten())
    # aoas           = list(rp.repeat(rp.arange(ncases), ncpts))
    # res            = list(rp.tile(rp.arange(ncpts), ncases))

    # X_BL_TOP_mask = X_BL_TOP_mask.at[first_panel, aoas, res].set(False)
    # X_BL_TOP      = rp.where(X_BL_TOP_mask, rp.nan, X_BL_TOP)

    # VE_BL_TOP = VT_BL_top
    # CP_BL_TOP = 1.0 - VE_BL_TOP**2

    # CP_BL_VALS = rp.concatenate([rp.flip(CP_BL_BOT, axis=0), CP_BL_TOP], axis=0)
    # CP_BL      = drop_nans_and_reshape_full(CP_BL_VALS)

    # DCP_DX = rp.diff(CP_BL, axis=0) / rp.diff(X_BL, axis=0)

    # AERO_RES = aero_coeff(x_coord_3d, y_coord_3d, CP, alpha, npanel)

    # del2_inf_l = THETA[0, :, :] * VE[0, :, :]**((5.0 + H[0, :, :]) / 2.0)
    # del2_inf_u = THETA[-1, :, :] * VE[-1, :, :]**((5.0 + H[-1, :, :]) / 2.0)
    # del2_inf   = del2_inf_u + del2_inf_l
    # cd_sqy     = 2.0 * del2_inf.T

    # fL, fD = chordwise_distribution(x_coord_3d, y_coord_3d, CP, alpha, npanel, CF, vt)

    # # Zero out wrong columns
    # AERO_RES.cl   = AERO_RES.cl.at[:, wrong_columns].set(0.0)
    # AERO_RES.cdpi = AERO_RES.cdpi.at[:, wrong_columns].set(0.0)
    # AERO_RES.cm   = AERO_RES.cm.at[:, wrong_columns].set(0.0)
    # cd_sqy        = cd_sqy.at[:, wrong_columns].set(0.0)
    # CP_BL         = CP_BL.at[:, wrong_columns, :].set(0.0)
    # DCP_DX        = DCP_DX.at[:, wrong_columns, :].set(0.0)
    # VE            = VE.at[:, wrong_columns, :].set(0.0)
    # DVE           = DVE.at[:, wrong_columns, :].set(0.0)
    # THETA         = THETA.at[:, wrong_columns, :].set(0.0)
    # DELTA_STAR    = DELTA_STAR.at[:, wrong_columns, :].set(0.0)
    # DELTA         = DELTA.at[:, wrong_columns, :].set(0.0)
    # RE_THETA      = RE_THETA.at[:, wrong_columns, :].set(0.0)
    # RE_X          = RE_X.at[:, wrong_columns, :].set(0.0)
    # H             = H.at[:, wrong_columns, :].set(0.0)
    # CF            = CF.at[:, wrong_columns, :].set(0.0)
    # fL            = fL.at[:, wrong_columns, :].set(0.0)
    # fD            = fD.at[:, wrong_columns, :].set(0.0)

    airfoil_properties_old = Data(
        AoA            = alpha,
        Re             = Re_L,
        cl_invisc      = AERO_RES.cl,  
        cd_invisc      = AERO_RES.cdpi, 
        cm_invisc      = AERO_RES.cm,
        cd_visc        = cd_sqy,
        normals        = rp.transpose(normals,(3,2,0,1)),
        x              = rp.transpose(X,(2,1,0)),
        y              = rp.transpose(Y,(2,1,0)),
        x_bl           = rp.transpose(X_BL ,(2,1,0)),
        y_bl           = rp.transpose(Y_BL ,(2,1,0)),
        cp             = rp.transpose(CP_BL,(2,1,0)),  
        dcp_dx         = rp.transpose(DCP_DX,(2,1,0)),            
        Ue_Vinf        = rp.transpose(VE   ,(2,1,0)),         
        dVe            = rp.transpose(DVE  ,(2,1,0)),   
        theta          = rp.transpose(THETA,(2,1,0)),      
        delta_star     = rp.transpose(DELTA_STAR,(2,1,0)),  
        delta          = rp.transpose(DELTA,(2,1,0)),  
        Re_theta       = rp.transpose(RE_THETA,(2,1,0)),  
        Re_x           = rp.transpose(RE_X,(2,1,0)),  
        H              = rp.transpose(H,(2,1,0)),            
        cf             = rp.transpose(CF,(2,1,0)),
        fL             = fL,
        fD             = fD
        )  

    return airfoil_properties_old


def concatenate_surfaces(X_BOT, X_TOP,
                         FUNC_BOT_SURF, FUNC_TOP_SURF,
                         npanel, ncases, ncpts,
                         wrong_columns):
    
    '''Interpolation of airfoil properties   
    
    Assumptions:
    None

    Source:
    None                                                                    
                                                                   
    Inputs:                                    
    X_BOT          - bottom surface of airfoil                                     [unitless]
    X_TOP          - top surface of airfoil                                        [unitless]
    FUNC_BOT_SURF  - airfoil property computation discretization on bottom surface [multiple units]
    FUNC_TOP_SURF  - airfoil property computation discretization on top surface    [multiple units]
    npanel         - number of panels                                              [unitless]
    ncases         - number of angle of attacks                                    [unitless]
    ncpts          - number of Reynolds numbers                                    [unitless]
                                                                 
    Outputs:                                           
    FUNC           - airfoil property in user specified discretization on entire
                     surface of airfoil                                            [multiple units]
      
    Properties Used:
    N/A  
    '''  
    
    FUNC = rp.zeros((npanel, ncases, ncpts))

    for case in range(ncases):
        if case in wrong_columns:
            continue

        for cpt in range(ncpts):
            bot_mask = ~rp.isnan(X_BOT[:, case, cpt])
            top_mask = ~rp.isnan(X_TOP[:, case, cpt])

            # bottom where valid, else top
            merged = rp.where(bot_mask,
                              FUNC_BOT_SURF[:, case, cpt],
                              FUNC_TOP_SURF[:, case, cpt])

            FUNC = FUNC.at[:, case, cpt].set(merged)

    return FUNC
