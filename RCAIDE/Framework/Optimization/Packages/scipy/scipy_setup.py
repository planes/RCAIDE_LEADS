# scipy_setup.py
# 
# Created:  Aug 2015, E. Botero 
# Modified: Feb 2017, M. Vegh
#           Mar 2020, E. Botero
#           Jul 2020, M. Clarke
#           May 2021, E. Botero 

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# rcaide imports
import RNUMPY as rp
import torch
from RCAIDE.Framework.Optimization.Packages.particle_swarm import particle_swarm_optimization 
from scipy.optimize import NonlinearConstraint
from RCAIDE.Framework.Optimization.Common import helper_functions as help_fun

# ----------------------------------------------------------------------
#  Something that should become a class at some point
# ---------------------------------------------------------------------- 
def SciPy_Solve(problem,solver='SLSQP', sense_step = 1.4901161193847656e-08, iter =200, tolerance = 1e-6, pop_size =  10 , prob_seed = None ):  
    """ This converts your RCAIDE Nexus problem into a SciPy optimization problem and solves it
        SciPy has many algorithms, they can be switched out by using the solver input. 

        Assumptions:
        1.4901161193847656e-08 is SLSQP default FD step in scipy

        Source:
        N/A

        Inputs:
        problem                   [nexus()]
        solver                    [str]
        sense_step                [float]

        Outputs:
        outputs                   [list]

        Properties Used:
        None
    """
    
    inp = problem.optimization_problem.inputs
    obj = problem.optimization_problem.objective
    con = problem.optimization_problem.constraints
    
    # Have the optimizer call the wrapper
    wrapper = lambda x:SciPy_Problem(problem,x)    
    
    # Set inputsq
    nam  = inp.name[:] # Names
    ini  = rp.array(inp.value[:,0],dtype=rp.float32) # Initials
    bndl = rp.array(inp.value[:,1],dtype=rp.float32) # Bounds
    bndu = rp.array(inp.value[:,2],dtype=rp.float32) # Bounds
    scl  = rp.array(inp.value[:,3],dtype=rp.float32) # Scale
    
    x   = ini/scl
    bnds = rp.zeros((len(inp.name),2))
    lb   = rp.zeros(len(inp.name))
    ub   = rp.zeros(len(inp.name))
    de_bnds = []    
    
    for ii in range(0,len(inp.name)):
        # Scaled bounds
        bnds = bnds.at[ii].set(rp.array([bndl[ii]/scl[ii],bndu[ii]/scl[ii]]))
        lb   = lb.at[ii].set(bndl[ii]/scl[ii])
        ub   = ub.at[ii].set(bndu[ii]/scl[ii])
        de_bnds.append((bndl[ii]/scl[ii],bndu[ii]/scl[ii]))  
     
    # Finalize problem statement and run
    if solver=='SLSQP':
        import torch
        with torch.autograd.detect_anomaly(check_nan=True):
            outputs = rp.scipy.optimize.fmin_slsqp(wrapper,x,f_eqcons=problem.equality_constraint,f_ieqcons=problem.inequality_constraint,bounds=bnds,\
                                                iter=iter, epsilon = sense_step, acc  = tolerance, full_output=True,  iprint=0)
    elif solver == 'differential_evolution':
        # Define constraints as a tuple of nonlinear constraints 
        scaled_constraints = []
        aliases            = problem.optimization_problem.aliases
        for ii in range(0,len(con)):
            de_constraint  = con[[ii]] 
            def fun(x):
                problem.evaluate(x)
                constraint_val = help_fun.get_values(problem,de_constraint,aliases)
                return rp.atleast_1d(constraint_val)
            
            bound  = help_fun.scale_const_bnds(con)
            if con.name_signs[ii][1]=='=':
                print('Nonlinear constraints for scipy differential evolution optimization has '
                      'the general inequality form. Consider rewriting equality constraint as two '
                      'separate inequality constraints')
            
            if con.name_signs[ii][1]=='>':
                nlc = NonlinearConstraint(fun,bound[ii], rp.inf) 
                
            elif con.name_signs[ii][1]=='<':
                nlc = NonlinearConstraint(fun, -rp.inf,bound[ii])
                
            scaled_constraints.append(nlc) 
            
        diff_evo_cons = tuple(scaled_constraints)    
        
        outputs = rp.scipy.optimize.differential_evolution(wrapper, bounds= de_bnds, strategy='best1bin', maxiter=1000, popsize = pop_size, \
                                                     tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=prob_seed, callback=None,\
                                                     disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate',\
                                                     workers=1,constraints=diff_evo_cons)
        
    elif solver == 'particle_swarm_optimization':
        outputs = particle_swarm_optimization(wrapper, lb, ub, f_ieqcons=problem.inequality_constraint, kwargs={}, swarmsize=pop_size ,\
                                              omega=0.5, phip=0.5, phig=0.5, maxiter=1000, minstep=1e-4, minfunc=1e-4, debug=False)    
    else:
        outputs = rp.scipy.optimize.minimize(wrapper,x,method=solver)
    
    return outputs
 
def SciPy_Problem(problem,x):
    """ This wrapper runs the RCAIDE problem and is called by the Scipy solver.
        Prints the inputs (x) as well as the objective value

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        problem   [nexus()]
        x         [array]

        Outputs:
        obj       [float]

        Properties Used:
        None
    """      
    
    print('Inputs')
    print(x)        
    obj   = problem.objective(x)
    print('Obj')
    print(obj)

    
    return obj

