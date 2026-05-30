# RCAIDE/Library/Missions/Common/Update/aerodynamics.py
# 
# 
# Created:  Jul 2023, M. Clarke 

# ----------------------------------------------------------------------------------------------------------------------
#  Imports
# ---------------------------------------------------------------------------------------------------------------------- 
import RNUMPY as rp

# ----------------------------------------------------------------------------------------------------------------------
#  Update Aerodynamics
# ---------------------------------------------------------------------------------------------------------------------- 
def aerodynamics(segment):
    """ Gets aerodynamics conditions
    
        Assumptions:
        +X out nose
        +Y out starboard wing
        +Z down

        Inputs:
            segment.analyses.aerodynamics_model                    [Function]
            aerodynamics_model.settings.maximum_lift_coefficient   [unitless]
            aerodynamics_model.vehicle.reference_area             [meter^2]
            segment.state.conditions.freestream.dynamic_pressure   [pascals]

        Outputs:
            conditions.aerodynamics.coefficients.lift.total [unitless]
            conditions.aerodynamics.coefficients.drag.total [unitless]
            conditions.frames.wind.force_vector [newtons]
            conditions.frames.wind.drag_force_vector [newtons]

        Properties Used:
        N/A
    """
    
    # unpack
    conditions         = segment.state.conditions
    q                  = segment.state.conditions.freestream.dynamic_pressure
    Sref               = segment.analyses.vehicle.reference_area
    MAC                = segment.analyses.vehicle.wings.main_wing.chords.mean_aerodynamic
    span               = segment.analyses.vehicle.wings.main_wing.spans.projected 
    aerodynamics_model = segment.analyses.aerodynamics
    CLmax              = aerodynamics_model.settings.maximum_lift_coefficient 
    
    # call aerodynamics model
    _ = aerodynamics_model(segment, segment.analyses.vehicle)     

    # Forces 
    CL = conditions.aerodynamics.coefficients.lift.total
    CD = conditions.aerodynamics.coefficients.drag.total
    CY = conditions.static_stability.coefficients.Y

    CL = rp.where(q <= 0.0, 0.0, CL)
    CD = rp.where(q <= 0.0, 0.0, CD)
    CL = rp.clip(CL, -CLmax, CLmax)

    # dimensionalize
    F      = segment.state.ones_row(3) * 0.0
    F = F.at[:,2].set(( -CL * q * Sref )[:,0])
    F = F.at[:,1].set((  CY * q * Sref  )[:,0])
    F = F.at[:,0].set(( -CD * q * Sref )[:,0])

    # rewrite aerodynamic CL and CD
    conditions.aerodynamics.coefficients.lift.total  = CL
    conditions.aerodynamics.coefficients.drag.total  = CD
    conditions.frames.wind.force_vector              = F

    # -----------------------------------------------------------------
    # Moments
    # -----------------------------------------------------------------
    C_L = conditions.static_stability.coefficients.L
    C_M = conditions.static_stability.coefficients.M
    C_N = conditions.static_stability.coefficients.N

    C_M = rp.where(q <= 0.0, 0.0, C_M)

    # dimensionalize
    M      = segment.state.ones_row(3) * 0.0
    M = M.at[:,0].set((C_L[:,0] * q[:,0] * Sref * span))
    M = M.at[:,1].set((C_M[:,0] * q[:,0] * Sref * MAC))
    M = M.at[:,2].set((C_N[:,0] * q[:,0] * Sref * span))

    # pack conditions
    conditions.frames.wind.moment_vector = M

    return