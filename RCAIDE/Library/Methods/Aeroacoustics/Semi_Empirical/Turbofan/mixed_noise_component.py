# RCAIDE/Methods/Aeroacoustics/Semi_Empirical/Engine/mixed_noise_component.py
# 
# 
# Created:  Jul 2023, M. Clarke  

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
 
# Python package imports   
import RNUMPY as rp   

# ----------------------------------------------------------------------------------------------------------------------     
#  Mixed Noise Component
# ----------------------------------------------------------------------------------------------------------------------       
def mixed_noise_component(Velocity_primary, theta_m, sound_ambient, Velocity_secondary,
                          Velocity_aircraft, Area_primary, Area_secondary, DSPL_m, EX_m, Str_m, Velocity_mixed, XBPR):
    """
    This function calculates the noise contribution of the mixed jet component.

    Parameters
    ----------
    Velocity_primary : float
        Velocity of the primary jet [m/s].
    theta_m : float
        Angle for the mixed jet [rad].
    sound_ambient : float
        Ambient sound level [SPL].
    Velocity_secondary : float
        Velocity of the secondary jet [m/s].
    Velocity_aircraft : float
        Velocity of the aircraft [m/s].
    Area_primary : float
        Area of the primary jet [m^2].
    Area_secondary : float
        Area of the secondary jet [m^2].
    DSPL_m : float
        Decibel Sound Pressure Level for the mixed jet [SPL].
    EX_m : float
        Excess noise level for the mixed jet.
    Str_m : float
        Strouhal number for the mixed jet.
    Velocity_mixed : float
        Velocity of the mixed jet [m/s].
    XBPR : float
        Bypass ratio adjustment factor.

    Returns
    -------
    SPL_m : float
        Sound Pressure Level for the mixed jet component [dB].

    Notes
    -----
    The function uses semi-empirical methods to calculate the noise contribution of the mixed jet component.

    **Definitions**

    'SPL_m'
        Sound Pressure Level for the mixed jet component.

    References
    ----------
    [1] SAE ARP876D: Gas Turbine Jet Exhaust Noise Prediction (original)
    [2] de Almeida, Odenir. "Semi-empirical methods for coaxial jet noise prediction." (2008). (adapted)
    """

    #Calculation of the velocity exponent
    velocity_exponent = (Velocity_mixed/sound_ambient)**0.5*(0.6+(0.2/(0.2+Str_m) * \
        rp.exp(-0.3*(theta_m+(Str_m/(1+Str_m))-2.7)**2)))

    #Calculation of the Source Strengh Function (FV)
    FV = ((Velocity_mixed-Velocity_aircraft)/sound_ambient)**velocity_exponent * \
        ((Velocity_mixed+Velocity_aircraft)/sound_ambient)**(1-velocity_exponent)

    #Determination of the noise model coefficients
    Z1 = -30*((1.8*theta_m/rp.pi)-0.6)**2
    Z2 = -9 -4*((Velocity_primary-Velocity_secondary)/sound_ambient)-38*((1.8*theta_m/rp.pi)-0.6)**3 + \
        30*(0.6-rp.log10(1+Area_secondary/Area_primary))*(1.8*theta_m/rp.pi - 0.6)
    Z3 = 1-0.4*((1.8*theta_m/rp.pi)-0.6)**2
    Z4 = 0.44-0.5/rp.exp(((4.5*theta_m/rp.pi)-4)**2) + 0.2*Velocity_primary/sound_ambient - \
        0.7*Velocity_mixed/sound_ambient - 0.2*rp.log10((1+Area_secondary)/Area_primary) + \
        0.05*(XBPR)*rp.exp(-5*(theta_m-2.4)**2)
    Z5 = 34 + 81*theta_m/rp.pi - 20*((1.8*theta_m/rp.pi)-0.6)**3
    Z6 = 108 + 37.8*theta_m/rp.pi + 5*Velocity_mixed*(Velocity_primary-Velocity_secondary)/(sound_ambient**2) - \
        rp.exp(-5*(theta_m-1.8)**2) + 7*Velocity_mixed/sound_ambient*(1-0.4*(Velocity_primary/sound_ambient) * \
        rp.exp(-0.7*rp.abs(Str_m-0.8))) / rp.exp(8*(theta_m-2.4)**2) + 0.8*(XBPR)*rp.exp(theta_m-2.3-Velocity_mixed/sound_ambient) + \
        DSPL_m + EX_m

    #Determination of Sound Pressure Level for the mixed jet component
    SPL_m = (Z1*rp.log10(FV)+Z2)*(rp.log10(Str_m)-Z3*rp.log10(FV)-Z4)**2 + Z5*rp.log10(FV) + Z6

    return SPL_m 
