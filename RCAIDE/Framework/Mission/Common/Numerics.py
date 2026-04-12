# RCAIDE/Framework/Analyses/Mission/Segments/Conditions/Numerics.py
# 
# 
# Created:  Jul 2023, M. Clarke

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from RCAIDE.Framework.Core import Data
from .Conditions import Conditions 
from RCAIDE.Library.Methods.Utilities.Chebyshev  import chebyshev_data 
import RNUMPY as rp

# ----------------------------------------------------------------------------------------------------------------------
#  Numerics
# ----------------------------------------------------------------------------------------------------------------------

class Numerics(Conditions):
    """ Creates the data structure for the numerical solving of a mission.
    
        Assumptions:
        None
        
        Source:
        None
    """
    
    def __defaults__(self):
        """This sets the default values.
    
            Assumptions:
            None
    
            Source:
            N/A
    
            Inputs:
            None
    
            Outputs:
            None
    
            Properties Used:
            None
        """           
        self.tag                              = 'numerics' 
        self.number_of_control_points         = 16
        self.discretization_method            = chebyshev_data
        self.solver                           = Conditions()
        self.solver.type                      = "optimize" # options: "optimize", "root_finder"
        self.solver.method                    = "SLSQP"    
        self.solver.objective                 = "energy"   # options: # None, energy , power 
        self.solver.tolerance_solution        = 1E-5     
        self.solver.converged                 = None
        self.solver.print_output              = True
        self.solver.max_evaluations           = 200
        self.solver.step_size                 = 1E-7
        self.solver.lower_bounds              = Conditions()
        self.solver.upper_bounds              = Conditions()
        
        self.dimensionless                    = Conditions()
        self.dimensionless.control_points     = rp.empty([0,0])
        self.dimensionless.differentiate      = rp.empty([0,0])
        self.dimensionless.integrate          = rp.empty([0,0]) 
            
        self.time                             = Conditions()
        self.time.control_points              = rp.empty([0,0])
        self.time.differentiate               = rp.empty([0,0])
        self.time.integrate                   = rp.empty([0,0]) 
        
        
        
        