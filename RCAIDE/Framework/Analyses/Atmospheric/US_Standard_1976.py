#RCAIDE/Frameworks/Analyses/Atmospheric/US_Standard_1976.py
#
# Created: Dec 2024, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
# RCAIDE imports 
import RCAIDE
from RCAIDE.Framework.Analyses.Atmospheric      import Atmospheric
from RCAIDE.Framework.Mission.Common.Conditions import Conditions 
from RCAIDE.Framework.Core.Arrays               import atleast_2d_col 
from RCAIDE.Library.Attributes.Gases            import Air
from RCAIDE.Library.Attributes.Planets          import Earth

# pthon imports 
import RNUMPY as rp
from warnings import warn 

# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------

class US_Standard_1976(Atmospheric):

    """ Implements the U.S. Standard Atmosphere (1976 version)
        
    Assumptions:
    None
    
    Source:
    U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976
    """
    
    def __defaults__(self):
        """This sets the default values for the analysis to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Output:
        None

        Properties Used:
        None
        """      
        atmo_data = RCAIDE.Library.Attributes.Atmospheres.Earth.US_Standard_1976()
        self.update(atmo_data)         
        planet = RCAIDE.Framework.Analyses.Planets.Earth()
        self.features.planet = planet.features
    
    def compute_values(self,altitude,temperature_deviation=0.0,var_gamma=False):

        """Computes atmospheric values.

        Assumptions:
        US 1976 Standard Atmosphere

        Source:
        U.S. Standard Atmosphere, 1976, U.S. Government Printing Office, Washington, D.C., 1976

        Inputs:
        altitude                                 [m]
        temperature_deviation                    [K]

        Output:
        atmo_data.
          pressure                               [Pa]
          temperature                            [K]
          speed_of_sound                         [m/s]
          dynamic_viscosity                      [kg/(m*s)]
          kinematic_viscosity                    [m^2/s]
          thermal_conductivity                   [W/(m*K)]
          prandtl_number                         [-]
           
        Properties Used:
        self.
          fluid_properties.gas_specific_constant [J/(kg*K)]
          planet.sea_level_gravity               [m/s^2]
          planet.mean_radius                     [m]
          breaks.
            altitude                             [m]
            temperature                          [K]
            pressure                             [Pa]
        """

        # unpack
        zs        = altitude
        gas       = self.fluid_properties
        planet    = self.planet
        grav      = self.planet.sea_level_gravity        
        Rad       = self.planet.mean_radius
        R         = gas.gas_specific_constant
        delta_isa = temperature_deviation
        
        # check properties
        if not gas == Air():
            warn('US Standard Atmosphere not using Air fluid properties')
        if not planet == Earth():
            warn('US Standard Atmosphere not using Earth planet properties')          
        
        # convert input if necessary
        zs = atleast_2d_col(zs)

        # get model altitude bounds
        zmin = self.breaks.altitude[0]
        zmax = self.breaks.altitude[-1]   
        
        # convert geometric to geopotential altitude
        zs = zs/(1 + zs/Rad)
        
        # check ranges
        if rp.amin(zs) < zmin:
            print("Warning: altitude requested below minimum for this atmospheric model; returning values for h = -2.0 km")
        if rp.amax(zs) > zmax:
            print("Warning: altitude requested above maximum for this atmospheric model; returning values for h = 86.0 km")   
        # Safely constrain zs between zmin and zmax in one pass
        zs = rp.clip(zs, min=zmin, max=zmax)
 

        # initialize return data
        zeros = rp.zeros_like(zs)
        p     = zeros * 0.0
        T     = zeros * 0.0
        rho   = zeros * 0.0
        a     = zeros * 0.0
        mu    = zeros * 0.0
        z0    = zeros * 0.0
        T0    = zeros * 0.0
        p0    = zeros * 0.0
        alpha = zeros * 0.0

        # populate the altitude breaks
        for i in range(len(self.breaks.altitude)-1): 
            i_inside = (zs >= self.breaks.altitude[i]) & (zs <= self.breaks.altitude[i+1])
            
            # Precompute the lapse rate for this layer
            alpha_val = -(self.breaks.temperature[i+1] - self.breaks.temperature[i]) / \
                         (self.breaks.altitude[i+1]    - self.breaks.altitude[i])
            
            z0    = rp.where(i_inside, self.breaks.altitude[i], z0)
            T0    = rp.where(i_inside, self.breaks.temperature[i], T0)
            p0    = rp.where(i_inside, self.breaks.pressure[i], p0)
            alpha = rp.where(i_inside, alpha_val, alpha)
        
        # Interpolate the breaks
        dz = zs - z0
        
        # Protect alpha for exponent and base calculation
        safe_alpha = rp.where(alpha == 0., 1.0, alpha)
        
        # Protect base of the power function
        base = 1. - alpha * dz / T0
        safe_base = rp.where(alpha == 0., 1.0, rp.maximum(base, 1e-10))
        
        # Calculate BOTH full arrays safely
        p_isoth = p0 * rp.exp(-1. * dz * grav / (R * T0))
        p_adiab = p0 * ( safe_base ** (1. * grav / (safe_alpha * R)) )
        
        # Select the correct result
        p = rp.where(alpha == 0., p_isoth, p_adiab)
        
        T     = T0 - dz*alpha + delta_isa
        rho   = gas.compute_density(T,p)
        a     = gas.compute_speed_of_sound(T,p,var_gamma)
        mu    = gas.compute_absolute_viscosity(T)
        K     = gas.compute_thermal_conductivity(T)  
        Pr    = gas.compute_prandtl_number(T)
        
        atmo_data = Conditions()
        atmo_data.expand_rows(zs.shape[0])
        atmo_data.pressure                     = p
        atmo_data.temperature                  = T
        atmo_data.density                      = rho
        atmo_data.speed_of_sound               = a
        atmo_data.dynamic_viscosity            = mu
        atmo_data.kinematic_viscosity          = mu/rho
        atmo_data.thermal_conductivity         = K
        atmo_data.prandtl_number               = Pr 
        
        return atmo_data