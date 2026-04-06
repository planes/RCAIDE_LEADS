# RCAIDE/Library/Methods/Stability/compute_dynamic_flight_modes.py
# 
# 
# Created:  Apr 2024, M. Clarke

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------

# RCAIDE imports
import RCAIDE 

# python imports 
import RNUMPY as rp 

# ----------------------------------------------------------------------------------------------------------------------
#  compute_dynamic_flight_modes
# ----------------------------------------------------------------------------------------------------------------------
def compute_dynamic_flight_modes(state,settings,vehicle): 
    """This function follows the stability axis EOM derivation in Blakelock
    to return the vehicle's dynamic modes and state space 
    
    Assumptions:
       Linerarized Equations are used following the reference below

    Source:
      Automatic Control of Aircraft and Missiles by J. Blakelock Pg 23 and 117
      Dynanmics of Flight Stability and Control by Bernard Edkin and Llyod Reid Chapter 5, 

    Inputs:
       conditions.aerodynamics  
       conditions.static_stability

    Outputs: 
       conditions.dynamic_stability.LatModes
       conditions.dynamic_stability.LongModes 

    Properties Used:
       N/A
     """

    conditions = state.conditions 
    AoA        = conditions.aerodynamics.angles.alpha  
    n_cpts     = state.numerics.number_of_control_points 
    S_ref      = vehicle.reference_area  
    c_ref      = vehicle.reference_chord
    b_ref      = vehicle.reference_span
 
    if (rp.count_nonzero(vehicle.mass_properties.moments_of_inertia.tensor) > 0) and (rp.all(rp.isnan(AoA)) !=  True):
        g      = conditions.freestream.gravity  
        rho    = conditions.freestream.density
        u0     = conditions.freestream.velocity
        qDyn0  = conditions.freestream.dynamic_pressure  
        theta0 = rp.arctan(conditions.frames.inertial.velocity_vector[:,2]/conditions.frames.inertial.velocity_vector[:,0])[:,None] 
        SS     = conditions.static_stability
        SSD    = SS.derivatives 
        DS     = conditions.dynamic_stability 
        Ixx    = conditions.weights.vehicle.moments_of_inertia_Ixx  
        Ixz    = conditions.weights.vehicle.moments_of_inertia_Ixz  
        Iyy    = conditions.weights.vehicle.moments_of_inertia_Iyy  
        Izx    = conditions.weights.vehicle.moments_of_inertia_Izx  
        Izz    = conditions.weights.vehicle.moments_of_inertia_Izz  
        m      = conditions.weights.vehicle.mass
        
        if rp.all(conditions.static_stability.spiral_criteria) == 0: 
            conditions.static_stability.spiral_criteria = SSD.CL_beta*SSD.CN_r / (SSD.CL_r*SSD.CN_beta) 
          
        ## Build longitudinal EOM A Matrix (stability axis)
        ALon = rp.zeros((n_cpts,4,4))
        BLon = rp.zeros((n_cpts,4,1))  
        

        # Elevator effectiveness 
        for wing in vehicle.wings:
            if isinstance(wing,RCAIDE.Library.Components.Wings.Horizontal_Tail): 
                horizontal_tail = wing  
            if isinstance(wing,RCAIDE.Library.Components.Wings.Main_Wing):  
                main_wing       = wing  
            if isinstance(wing,RCAIDE.Library.Components.Wings.Blended_Wing_Body): 
                main_wing       = wing
                horizontal_tail = wing 
        
        # unpack unit conversions  
        a_t              = 2 * rp.pi # dCL_t_dalphat 
        l_t              = (horizontal_tail.origin[0][0] +horizontal_tail.aerodynamic_center[0]) - vehicle.mass_properties.center_of_gravity[0][0] # disstance from CG to tail AC
        S_t              = horizontal_tail.areas.reference # tail area
        S                = main_wing.areas.reference # wing area
        l_bar_t          = (horizontal_tail.origin[0][0] +  horizontal_tail.aerodynamic_center[0]) -(main_wing.origin[0][0] +  main_wing.aerodynamic_center[0])  # distance from AC of main wing to tail AC
        V_H              = ( l_bar_t *S_t ) /(c_ref *S)  # tail volume
        dEpsilon_dalpha  =  0.3
        SSD.CZ_alpha_dot =  a_t * dEpsilon_dalpha *  (l_t /u0) *  (S_t / S) 
        SSD.CM_alpha_dot =  -a_t * V_H * dEpsilon_dalpha*  (l_t /u0)    
        Cw               = m * g / (qDyn0 * S_ref)  
        Xu               = rho * u0 * S_ref * Cw * rp.sin(theta0) + 0.5 * rho * u0 * S_ref * SSD.CX_u  
        Xw               = 0.5 * rho * u0 * S_ref * SSD.CX_alpha     
        Zu               = -rho * u0 * S_ref * Cw * rp.cos(theta0) + 0.5 * rho * u0 * S_ref * SSD.CZ_u 
        Zw               = 0.5 * rho * u0 * S_ref * SSD.CZ_alpha   
        Zq               = 0.25 * rho * u0 * c_ref * S_ref *  SSD.CZ_q    
        Mu               =  0.5 * rho * u0 * c_ref * S_ref * SSD.CM_u     
        Mw               = 0.5 * rho * u0 * c_ref * S_ref * SSD.CM_alpha  
        Mq               = 0.25 * rho * u0 * c_ref * c_ref * S_ref * SSD.CM_q     
        ZwDot            = 0.25 * rho * c_ref * S_ref * SSD.CZ_alpha_dot  
        MwDot            = 0.25 * rho * c_ref * S_ref * SSD.CM_alpha_dot  
        
        
        ALon[:,0,0] = (Xu / m).T[0]
        ALon[:,0,1] = (Xw / m).T[0] 
        ALon[:,0,3] = (-g * rp.cos(theta0)).T[0]
        ALon[:,1,0] = (Zu / (m - ZwDot)).T[0]
        ALon[:,1,1] = (Zw / (m - ZwDot)).T[0] 
        ALon[:,1,2] = ((Zq + (m * u0)) / (m - ZwDot) ).T[0]
        ALon[:,2,0] = ((Mu + MwDot * Zu / (m - ZwDot)) / Iyy).T[0]  
        ALon[:,2,1] = ((Mw + MwDot * Zw / (m - ZwDot)) / Iyy).T[0] 
        ALon[:,2,2] = ((Mq + MwDot * (Zq + m * u0) / (m - ZwDot)) / Iyy ).T[0] 
        ALon[:,2,3] = (-MwDot * m * g * rp.sin(theta0) / (Iyy * (m - ZwDot))).T[0] 
        ALon[:,3,0] = 0
        ALon[:,3,1] = 0
        ALon[:,3,2] = 1
        ALon[:,3,3] = 0 
 
        ele = conditions.control_surfaces.elevator.static_stability.coefficients 
        Xe  = 0 # Neglect
        Ze  = 0.5 * rho * u0 * u0 * S_ref * ele.lift
        Me  = 0.5 * rho * u0 * u0 * S_ref * c_ref * ele.M
        
        BLon[:,0,0] = Xe / m[:, 0]
        BLon[:,1,0] = (Ze / (m - ZwDot)).T[0]
        BLon[:,2,0] = (Me / Iyy + MwDot / Iyy * Ze / (m - ZwDot)).T[0]
        BLon[:,3,0] = 0        

        # Look at eigenvalues and eigenvectors
        LonModes                  = rp.zeros((n_cpts,4), dtype = complex)
        phugoidFreqHz             = rp.zeros((n_cpts,1))
        phugoidDamping            = rp.zeros((n_cpts,1))
        phugoidTimeDoubleHalf     = rp.zeros((n_cpts,1))
        shortPeriodFreqHz         = rp.zeros((n_cpts,1))
        shortPeriodDamping        = rp.zeros((n_cpts,1))
        shortPeriodTimeDoubleHalf = rp.zeros((n_cpts,1)) 
        try: 
            LonModes  , V = rp.linalg.eig(ALon)  
            phugoidInd                    = rp.argmax(LonModes,axis=1)
            Ind                           = rp.arange(n_cpts)
            phugoidFreqHz                 = rp.atleast_2d(abs(LonModes[Ind, phugoidInd]) / (2 * rp.pi)).T
            phugoidDamping                = rp.atleast_2d(rp.sqrt(1/ (1 + ( LonModes[Ind,phugoidInd].imag/ LonModes[Ind,phugoidInd].real )**2 ))).T 
            phugoidTimeDoubleHalf         = rp.log(2) / abs(2 * rp.pi * phugoidFreqHz * phugoidDamping)
            
            # Find short period
            shortPeriodInd               = rp.argmin(LonModes,axis=1)
            shortPeriodFreqHz            = rp.atleast_2d(abs(LonModes[Ind, shortPeriodInd]) / (2 * rp.pi)).T
            shortPeriodDamping           = rp.atleast_2d(rp.sqrt(1/ (1 + (LonModes[Ind, shortPeriodInd].imag/LonModes[Ind,shortPeriodInd].real)**2 ))).T
            shortPeriodTimeDoubleHalf    = rp.log(2) / abs(2 * rp.pi * shortPeriodFreqHz * shortPeriodDamping)
        except:
            pass
        
        ## Build lateral EOM A Matrix (stability axis)
        ALat = rp.zeros((n_cpts,4,4))
        BLat = rp.zeros((n_cpts,4,1))
        
        # Need to compute Ixx, Izz, and Ixz as a function of alpha. Which alpha? I wouldn't expect this to change.
        R    = rp.zeros((n_cpts,2,2))
        modI = rp.zeros((n_cpts,2,2)) 

        R[:,0, 0]  = rp.cos(AoA[:, 0])
        R[:,0, 1]  = - rp.sin(AoA[:, 0])
        R[:,1, 0]  = rp.sin( AoA[:, 0])
        R[:,1, 1]  = rp.cos(AoA[:, 0])
         
        modI[:,0, 0]    = Ixx[:, 0]
        modI[:,0, 1]    = Ixz[:, 0]
        modI[:,1, 0]    = Izx[:, 0]
        modI[:,1, 1]    = Izz[:, 0]
                               
        INew      = R * modI * rp.transpose(R, axes = (0,2,1))
        IxxStab   =  INew[:,0,0]
        IxzStab   = -INew[:,0,1]
        IzzStab   =  INew[:,1,1]
        Ixp       = rp.atleast_2d((IxxStab * IzzStab - IxzStab**2) / IzzStab).T
        Izp       = rp.atleast_2d((IxxStab * IzzStab - IxzStab**2) / IxxStab).T
        Ixzp      = rp.atleast_2d(IxzStab / (IxxStab * IzzStab - IxzStab**2)).T 
            
        Yv = 0.5 * rho * u0 * S_ref * SSD.CY_beta 
        Yr = 0.25 * rho * u0 * b_ref * S_ref * SSD.CY_r
        Lv = 0.5 * rho * u0 * b_ref * S_ref * SSD.CL_beta
        Lp = 0.25 * rho * u0 * b_ref**2 * S_ref * SSD.CL_p
        Lr = 0.25 * rho * u0 * b_ref**2 * S_ref * SSD.CL_r
        Nv = 0.5 * rho * u0 * b_ref * S_ref * SSD.CN_beta
        Np = 0.25 * rho * u0 * b_ref**2 * S_ref * SSD.CN_p
        Nr = 0.25 * rho * u0 * b_ref**2 * S_ref * SSD.CN_r
        
        # Aileron effectiveness  
        # WHat is we don't have ailerons? Will this work? Check this
        ail = conditions.control_surfaces.aileron.static_stability.coefficients                  
        Ya = 0.5 * rho * u0 * u0 * S_ref * ail.Y 
        La = 0.5 * rho * u0 * u0 * S_ref * b_ref * ail.L 
        Na = 0.5 * rho * u0 * u0 * S_ref * b_ref * ail.N 
        
        BLat[:,0,0] = (Ya / m).T[0]
        BLat[:,1,0] = (La / Ixp + Ixzp * Na).T[0]
        BLat[:,2,0] = (Ixzp * La + Na / Izp).T[0]
        BLat[:,3,0] = 0
     
        ALat[:,0,0] = (Yv / m).T[0]  
        ALat[:,0,2] = (Yr/m - u0).T[0] 
        ALat[:,0,3] = (g * rp.cos(theta0)).T[0]
        
        ALat[:,1,0] = (Lv / Ixp + Ixzp * Nv).T[0] 
        ALat[:,1,1] = (Lp / Ixp + Ixzp * Np).T[0] 
        ALat[:,1,2] = (Lr / Ixp + Ixzp * Nr).T[0] 
        ALat[:,1,3] = 0
        
        ALat[:,2,0] = (Ixzp * Lv + Nv / Izp).T[0] 
        ALat[:,2,1] = (Ixzp * Lp + Np / Izp).T[0] 
        ALat[:,2,2] = (Ixzp * Lr + Nr / Izp).T[0] 
        ALat[:,2,3] = 0
        
        ALat[:,3,0] = 0
        ALat[:,3,1] = 1
        ALat[:,3,2] = (rp.tan(theta0)).T[0] 
        ALat[:,3,3] = 0
                                    
        LatModes                    = rp.zeros((n_cpts,4),dtype=complex)
        dutchRollFreqHz             = rp.zeros((n_cpts,1))
        dutchRollDamping            = rp.zeros((n_cpts,1))
        dutchRollTimeDoubleHalf     = rp.zeros((n_cpts,1))
        rollSubsistenceFreqHz       = rp.zeros((n_cpts,1))
        rollSubsistenceTimeConstant = rp.zeros((n_cpts,1))
        rollSubsistenceDamping      = rp.zeros((n_cpts,1))
        spiralFreqHz                = rp.zeros((n_cpts,1))
        spiralTimeDoubleHalf        = rp.zeros((n_cpts,1))
        spiralDamping               = rp.zeros((n_cpts,1))
        dutchRoll_mode_real         = rp.zeros((n_cpts,1))
         
        try: 
            LatModes  , V = rp.linalg.eig(ALat) # State order: u, w, q, theta
            # Change to LatModes? If we changed D above then mind as well change this too
            # LatModes[:,:] =  D[:,:]  
            
            real_parts = LatModes.real
            unique_elements, counts = rp.unique(real_parts, return_counts=True, axis=1)
            idx = rp.where(counts==2)[0]
    
            dutchRollFreqHz         = abs(LatModes[:,idx]) / (2 * rp.pi)
            dutchRollDamping        = rp.sqrt(1/ (1 + ( LatModes[:,idx].imag/ LatModes[:,idx].real )**2 ))  
            dutchRollTimeDoubleHalf = rp.log(2) / abs(2 * rp.pi * dutchRollFreqHz * dutchRollDamping)
            dutchRoll_mode_real     = LatModes[:,idx].real / (2 * rp.pi)
             
            dutch_roll_idx                = rp.where( unique_elements[:, idx] != LatModes.real ) 
            remaining_modes               = LatModes[dutch_roll_idx].reshape(n_cpts,2)
            rollInd                       = rp.argmin(remaining_modes,axis=1)
            rollSubsistenceFreqHz         = rp.atleast_2d(abs(remaining_modes[Ind,rollInd]) / 2 / rp.pi).T
            rollSubsistenceDamping        =  rp.atleast_2d(- rp.sign(remaining_modes[Ind,rollInd].real)).T
            rollSubsistenceTimeConstant   = 1 / (2 * rp.pi * rollSubsistenceFreqHz  * rollSubsistenceDamping)
            
            # Find spiral mode 
            spiralInd                   = rp.argmax(remaining_modes,axis=1)           
            spiralFreqHz                = rp.atleast_2d(abs(remaining_modes[Ind,spiralInd]) / 2 / rp.pi).T
            spiralDamping               = rp.atleast_2d(- rp.sign(remaining_modes[Ind,spiralInd].real)).T
            spiralTimeDoubleHalf        = rp.log(2) / abs(2 * rp.pi * spiralFreqHz  * spiralDamping)
        except:
            pass 
        
        # Inertial coupling susceptibility
        # See Etkin & Reid pg. 118 
        DS.pMax = min(min(rp.sqrt(-Mw * u0 / (Izz - Ixx))), min(rp.sqrt(-Nv * u0 / (Iyy - Ixx)))) 
        
        # -----------------------------------------------------------------------------------------------------------------------  
        # Store Results
        # ------------------------------------------------------------------------------------------------------------------------  
        DS.LongModes.LongModes                    = LonModes
        DS.LongModes.phugoidFreqHz                = phugoidFreqHz
        DS.LongModes.phugoidDamping               = phugoidDamping
        DS.LongModes.phugoidTimeDoubleHalf        = phugoidTimeDoubleHalf
        DS.LongModes.shortPeriodFreqHz            = shortPeriodFreqHz
        DS.LongModes.shortPeriodDamping           = shortPeriodDamping
        DS.LongModes.shortPeriodTimeDoubleHalf    = shortPeriodTimeDoubleHalf
                                                                        
        DS.LatModes.LatModes                      = LatModes   
        DS.LatModes.dutchRollFreqHz               = dutchRollFreqHz
        DS.LatModes.dutchRollDamping              = dutchRollDamping
        DS.LatModes.dutchRollTimeDoubleHalf       = dutchRollTimeDoubleHalf
        DS.LatModes.dutchRoll_mode_real           = dutchRoll_mode_real 
        DS.LatModes.rollSubsistenceFreqHz         = rollSubsistenceFreqHz
        DS.LatModes.rollSubsistenceTimeConstant   = rollSubsistenceTimeConstant
        DS.LatModes.rollSubsistenceDamping        = rollSubsistenceDamping
        DS.LatModes.spiralFreqHz                  = spiralFreqHz
        DS.LatModes.spiralTimeDoubleHalf          = spiralTimeDoubleHalf 
        DS.LatModes.spiralDamping                 = spiralDamping
    
    return 
