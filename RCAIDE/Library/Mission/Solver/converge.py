# RCAIDE/Library/Missions/Segments/converge.py
# 
# 
# Created:  Jul 2023, M. Clarke  

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
import RCAIDE
from RCAIDE.Framework.Core import  Units, Data
from RCAIDE.Framework.Optimization.Packages.scipy import scipy_setup
from RCAIDE.Framework.Optimization.Common         import Nexus
from RCAIDE.Framework.Analyses.Process            import Process

import RNUMPY as rp 

import numpy as np
import sys 
import os 


# ----------------------------------------------------------------------------------------------------------------------
# converge root
# ---------------------------------------------------------------------------------------------------------------------- 
def converge(segment):
    """Interfaces the mission a root finder algorithm.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    segment                            [Data]
    segment.settings.root_finder       [Data]
    state.numerics.tolerance_solution  [Unitless]

    Outputs:
    state.unknowns                     [Any]
    segment.state.numerics.converged   [Unitless]

    Properties Used:
    N/A
    """ 
    
    if segment.state.numerics.solver.type  == "optimize": 
        problem  = add_mission_variables(segment) 
       
        # Commense suppression of console window output  
        devnull = open(os.devnull,'w')
        sys.stdout = devnull
         
        outputs  = scipy_setup.SciPy_Solve(problem,
                                           solver     = segment.state.numerics.solver.method,
                                           sense_step = segment.state.numerics.solver.step_size,
                                           iter       = segment.state.numerics.solver.max_evaluations,
                                           tolerance  = segment.state.numerics.solver.tolerance_solution)
    
        # Terminate suppression of console window output   
        sys.stdout = sys.__stdout__  
         
        if outputs[3] != 0:
            mission_converge = False        
            error_message =  outputs[4] 
        else:
            mission_converge = True
     
    elif segment.state.numerics.solver.type  == "root_finder": 
        unknowns = segment.state.unknowns.pack_array() 
         
        if segment.state.number_of_unknowns != segment.state.number_of_residuals:
            raise AttributeError('\n The system of equations representing the mission is not square. The number of unknowns (' + str(segment.state.number_of_unknowns) + \
                                 ') is not equal to the number of residuals (equations) (' + str(segment.state.number_of_residuals) + '). Either enforce of unknowns '+\
                                 ' to be equal to the number of residuals (equations) to use fsolve or switch RCAIDE solver type to "optimize" when defining the segment.'+ \
                                 '\n i.e. segment.state.numerics.solver.type  = "optimize" ') 
        else:
            unknowns,infodict,ier,error_message = rp.scipy.optimize.fsolve(iterate_root_finder,
                                                 unknowns,
                                                 args   = segment,
                                                 xtol   = segment.state.numerics.solver.tolerance_solution,
                                                 maxfev = segment.state.numerics.solver.max_evaluations,
                                                 epsfcn = segment.state.numerics.solver.step_size,
                                                 full_output = 1)
            
            # Run the mission again to reset the unknowns to the final ones
            residuals = iterate_root_finder(unknowns,segment)
        
        if ier !=1:
            mission_converge = False
        else:
            mission_converge = True
            
    else: 
        raise Exception('undefined mission solver type')        
        
    if mission_converge == False:
        print("Segment did not converge. Segment Tag: " + segment.tag)
        print("Error Message:\n" + error_message)
        segment.state.numerics.solver.converged = False
        segment.converged = False
    else:
        segment.state.numerics.solver.converged = True
        segment.converged = True
                                
    return
    
# ---------------------------------------------------------------------------------------------------------------------- 
#  Helper Functions
# ---------------------------------------------------------------------------------------------------------------------- 
def iterate_root_finder(unknowns, segment):
    
    """Runs one iteration of of all analyses for the mission.

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    state.unknowns                [Data]
    segment.process.iterate       [Data]

    Outputs:
    residuals                     [Unitless]

    Properties Used:
    N/A
    """       
    if isinstance(unknowns,rp.ndarray):
        segment.state.unknowns.unpack_array(unknowns)
    else:
        segment.state.unknowns = unknowns
        
    segment.process.iterate(segment)
    
    residuals = segment.state.residuals.pack_array()
        
    return residuals



def add_mission_variables(segment):
    """Make a pretty table view of the problem with objective and constraints at the current inputs for the dummy solver
    

        Assumptions:
        N/A

        Source:
        N/A

        Inputs:
        x                  [vector]

        Outputs:
        input              [array]
        const_table        [array]

        Properties Used:
        None
    """             
    
    # Step 1: Define Nexus
    nexus                        = Nexus()
    optimization_problem         = Data() 
    
    # Step 2 : Get segment type 
    ground_seg_flag =  (type(segment) == RCAIDE.Framework.Mission.Segments.Ground.Landing) or\
                       (type(segment) == RCAIDE.Framework.Mission.Segments.Ground.Takeoff) or \
                       (type(segment) == RCAIDE.Framework.Mission.Segments.Ground.Ground)  
    single_pt_seg = (type(segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude) or\
                    (type(segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_AVL_Trimmed) or \
                    (type(segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_No_Propulsion) or \
                    (type(segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Throttle)  
    
    # Step 2: Optimizer Inputs 
    # Step 2.1: Extract inputs
    input_count   = 0
    unknown_keys  = list(segment.state.unknowns.keys())  
    unknown_keys.remove('tag') 
    if ground_seg_flag: 
        n_points      = segment.state.numerics.number_of_control_points
        len_inputs    = n_points
        len_residuals = n_points
    elif single_pt_seg:
        n_points      = 1
        len_inputs    = segment.state.number_of_unknowns
        len_residuals = segment.state.number_of_residuals
    else:
        n_points      = segment.state.numerics.number_of_control_points  
        len_inputs    = n_points*segment.state.number_of_unknowns
        len_residuals = n_points*segment.state.number_of_residuals
          
            
    full_unkn_vals        = Data()
    full_upper_bound_vals = Data()
    full_lower_bound_vals = Data()
    for unkn in unknown_keys: 
        full_unkn_vals[unkn]  = segment.state.unknowns[unkn]
        full_lower_bound_vals[unkn] = rp.atleast_2d(segment.state.numerics.solver.lower_bounds[unkn])
        full_upper_bound_vals[unkn] = rp.atleast_2d(segment.state.numerics.solver.upper_bounds[unkn])

    # Step 2.2: Construct nexus format  : [Variable_###, initial, -rp.inf, rp.inf , scaling, Units.less]
    initial_values    = full_unkn_vals.pack_array()
    input_len_strings = np.tile('Variable_', len_inputs)
    input_numbers     = rp.linspace(1,len_inputs,len_inputs,dtype=rp.int16)
    input_names       = np.core.defchararray.add(input_len_strings,np.array(input_numbers+input_count).astype(str))
    lower_bounds      = full_lower_bound_vals.pack_array()
    upper_bounds      = full_upper_bound_vals.pack_array()     
    units             = rp.broadcast_to(rp.array(Units.less),(len_inputs,))
    # scaling factor for optimizer 
    factor = rp.ceil(rp.log10(abs(initial_values)))
    factor = rp.where(rp.isinf(factor), 0, factor)
    scale  = 10 ** (factor)
    
    # Step 2.4 Add in the inputs 
    optimization_problem.inputs = Data()
    optimization_problem.inputs.name = input_names
    
    new_inputs_value = rp.zeros((len_inputs, 5))
    new_inputs_value = new_inputs_value.at[:,0].set(rp.ravel(initial_values))
    new_inputs_value = new_inputs_value.at[:,1].set(rp.ravel(lower_bounds))
    new_inputs_value = new_inputs_value.at[:,2].set(rp.ravel(upper_bounds))
    new_inputs_value = new_inputs_value.at[:,3].set(rp.ravel(scale))
    new_inputs_value = new_inputs_value.at[:,4].set(rp.ravel(units))
    optimization_problem.inputs.value = rp.array(new_inputs_value, dtype=rp.float32)
    
    # Step 3: Constraints 
    # Step 3.1 : Create the equality constraints to the beginning of the constraints all equality constraints are 0, scale 1, and unitless
    con_count       = 0
    con_len_strings = np.tile('Residual_', len_residuals)
    con_numbers     = rp.linspace(1,len_residuals,len_residuals,dtype=rp.int16)
    con_names       = np.core.defchararray.add(con_len_strings,np.array(con_numbers+con_count).astype(str))
    equals          = np.broadcast_to('=',(len_residuals,))
    zeros           = rp.zeros(len_residuals)
    ones            = rp.ones(len_residuals)
    
    # Step 3.2 Add in the new constraints
    optimization_problem.constraints = Data()
    new_con_name_signs = np.empty((len_residuals, 2), dtype=object)
    new_con_name_signs[:,0] = con_names
    new_con_name_signs[:,1] = equals
    optimization_problem.constraints.name_signs = new_con_name_signs
    
    new_con_value = rp.zeros((len_residuals, 3))
    new_con_value = new_con_value.at[:,0].set(zeros)
    new_con_value = new_con_value.at[:,1].set(ones)
    new_con_value = new_con_value.at[:,2].set(1*Units.less)
    optimization_problem.constraints.value = rp.array(new_con_value, dtype=rp.float32)            
    
    # Step 4. Aliases 
    # Step 4.1: Setup the aliases for the inputs
    basic_string_con = Data()
    input_string = []

    if ground_seg_flag:       
        output_numbers = rp.linspace(0,n_points-2,n_points-1,dtype=rp.int16)
        basic_string_con[unknown_keys[1]] = np.tile('segment.state.unknowns.'+unknown_keys[1]+'[', n_points-1)
        input_string.append(np.core.defchararray.add(basic_string_con[unknown_keys[1]],np.array(output_numbers).astype(str)))
        input_string        = np.array(input_string[0])
        input_string        = np.core.defchararray.add(input_string, rp.tile(']',len_inputs-1))
        input_aliases       = np.reshape(rp.tile(rp.atleast_2d(np.array((None,None))),len_inputs), (-1, 2)) 
        input_aliases[:,0]  = input_names
        input_aliases[0,1]  = 'segment.state.unknowns.'+unknown_keys[0] 
        input_aliases[1:,1] = input_string 
        
    elif single_pt_seg:  
        for unkn in unknown_keys:
            basic_string_con[unkn] = rp.tile('segment.state.unknowns.'+unkn+'[', n_points)
            input_string.append(np.core.defchararray.add(basic_string_con[unkn],np.array([0]).astype(str)))
        input_string       = np.ravel(input_string)
        input_string       = np.core.defchararray.add(input_string, np.tile(']',len_inputs))
        input_aliases      = np.reshape(rp.tile(rp.atleast_2d(np.array((None,None))),len_inputs), (-1, 2)) 
        input_aliases[:,0] = input_names
        input_aliases[:,1] = input_string
    else:  
        output_numbers = rp.linspace(0,n_points-1,n_points,dtype=rp.int16) 
        for unkn in unknown_keys:
            basic_string_con[unkn] = np.tile('segment.state.unknowns.'+unkn+'[', n_points)
            input_string.append(np.core.defchararray.add(basic_string_con[unkn],np.array(output_numbers).astype(str)))
        input_string       = np.ravel(input_string)
        input_string       = np.core.defchararray.add(input_string, np.tile(']',len_inputs))
        input_aliases      = np.reshape(np.tile(np.atleast_2d(np.array((None,None))),len_inputs), (-1, 2)) 
        input_aliases[:,0] = input_names
        input_aliases[:,1] = input_string
    
    # Step 4.2: Setup the aliases for the residuals
    basic_string_res      = np.tile('segment.state.residuals.pack_array()[', len_residuals)
    residual_string       = np.core.defchararray.add(basic_string_res,np.array(con_numbers-1).astype(str))
    residual_string       = np.core.defchararray.add(residual_string, np.tile(']',len_residuals))
    residual_aliases      = np.reshape(np.tile(np.atleast_2d(np.array((None,None))),len_residuals), (-1, 2)) 
    residual_aliases[:,0] = con_names
    residual_aliases[:,1] = residual_string
        
    # Step 4.3: Append Aliases
    aliases = []
    for ii in range(len_inputs):
        aliases.append(input_aliases[ii].tolist())
    for jj in range(len_residuals):   
        aliases.append(residual_aliases[jj].tolist())
    
    # Step 5: Objective function
    if segment.state.numerics.solver.objective == None:     
        aliases.append([ 'nothing'                   , 'postprocess.nothing']) 
        optimization_problem.objective = Data()
        optimization_problem.objective.name = np.array([['nothing']], dtype=object)
        optimization_problem.objective.value = rp.array([[1, 1*Units.less]], dtype=rp.float32)
    elif segment.state.numerics.solver.objective == "energy":
        aliases.append([ 'energy_consumed'          , 'postprocess.energy_consumed']) 
        optimization_problem.objective = Data()
        optimization_problem.objective.name = np.array([['energy_consumed']], dtype=object)
        optimization_problem.objective.value = rp.array([[1, 1*Units.less]], dtype=rp.float32)
    elif segment.state.numerics.solver.objective == "power":
        aliases.append([ 'maximum_power'          , 'postprocess.maximum_power'])
        optimization_problem.objective = Data()
        optimization_problem.objective.name = np.array([['maximum_power']], dtype=object)
        optimization_problem.objective.value = rp.array([[1, 1*Units.less]], dtype=rp.float32)
    else:
        raise Exception('undefined objective function')
    
    # append aliases 
    optimization_problem.aliases = aliases        
    
    # Step 6: Expand Rows  
    segment.process.initialize.expand_state(segment)
    
    # Step 7: Update iteration
    input_count = input_count+input_numbers[-1]      
     
    # Step 8: Append segment
    nexus.segment = segment
     
    # Step 9: Append procedure
    nexus.procedure = iterate_segment()
     
    # Step 9: Append post-process
    nexus.postprocess = Data()
    
    # Step 10: Append optimization problem 
    nexus.optimization_problem   = convert_problem_style(optimization_problem)


    
    return nexus

def convert_problem_style(problem):
    # 1. INPUTS
    if hasattr(problem, 'inputs') and not isinstance(problem.inputs, Data):
        arr = rp.array(problem.inputs or [], dtype=object)
        if len(arr) > 0 and arr.ndim == 1:
            arr = arr.reshape(1, -1) # Force 2D if passed a flat list
            
        D = Data()
        if len(arr) == 0:
            D.name = rp.array([], dtype=object).reshape(0, 1)
            D.value = rp.array([], dtype=rp.float).reshape(0, 0)
        else:
            D.name = arr[:, 0:1] # 0:1 keeps it 2D
            D.value = rp.array(arr[:, 1:], dtype=rp.float)
        problem.inputs = D

    # 2. OBJECTIVE
    if hasattr(problem, 'objective') and not isinstance(problem.objective, Data):
        arr = rp.array(problem.objective or [], dtype=object)
        if len(arr) > 0 and arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        D = Data()
        if len(arr) == 0:
            D.name = rp.array([], dtype=object).reshape(0, 1)
            D.value = rp.array([], dtype=rp.float).reshape(0, 0)
        else:
            D.name = arr[:, 0:1]
            D.value = rp.array(arr[:, 1:], dtype=rp.float)
        problem.objective = D

    # 3. CONSTRAINTS
    if hasattr(problem, 'constraints') and not isinstance(problem.constraints, Data):
        arr = rp.array(problem.constraints or [], dtype=object)
        if len(arr) > 0 and arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        C = Data()
        if len(arr) == 0:
            C.name_signs = rp.array([], dtype=object).reshape(0, 2)
            C.value = rp.array([], dtype=rp.float).reshape(0, 0)
        else:
            C.name_signs = arr[:, 0:2]
            C.value = rp.array(arr[:, 2:], dtype=rp.float)
        problem.constraints = C

    # 4. OUTPUTS (Adding this since your debugger referenced it)
    if hasattr(problem, 'outputs') and not isinstance(problem.outputs, Data):
        arr = rp.array(problem.outputs or [], dtype=object)
        if len(arr) > 0 and arr.ndim == 1:
            arr = arr.reshape(1, -1)
            
        O = Data()
        if len(arr) == 0:
            O.name = rp.array([], dtype=object).reshape(0, 1)
            O.value = rp.array([], dtype=rp.float).reshape(0, 0)
        else:
            O.name = arr[:, 0:1]
            O.value = rp.array(arr[:, 1:], dtype=rp.float)
        problem.outputs = O

    return problem

    return problem

def iterate_segment(): 
    procedure                           = Process()  
    procedure.segment                   = Process()
    procedure.segment.design_mission    = iterate_optimizer     
    procedure.post_process              = segment_post_process   
        
    return procedure
    
def iterate_optimizer(nexus):
    segment = nexus.segment
     
    unknowns = segment.state.unknowns.pack_array()
    if isinstance(unknowns,rp.ndarray):
        segment.state.unknowns.unpack_array(unknowns)
    else:
        segment.state.unknowns = unknowns
        
    segment.process.iterate(segment)
    
    residuals = segment.state.residuals.pack_array()    
    nexus.residuals =  residuals
    return nexus


  
def segment_post_process(nexus):
    # unpack
    power      = nexus.segment.state.conditions.energy.power
    I          = nexus.segment.state.numerics.time.integrate
    
    # compute max power of segment 
    max_power  = rp.max(nexus.segment.state.conditions.energy.power)
    
    # compute total energy consumed 
    if (type(nexus.segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude) or\
                    (type(nexus.segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_AVL_Trimmed) or \
                    (type(nexus.segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Altitude_No_Propulsion) or \
                    (type(nexus.segment) == RCAIDE.Framework.Mission.Segments.Single_Point.Set_Speed_Set_Throttle): 
        energy_consumed =  0
    else:
        energy_consumed = rp.dot(I,power)[-1][0]
    
    postprocess                 = nexus.postprocess
    postprocess.maximum_power   = max_power
    postprocess.energy_consumed = energy_consumed 
    postprocess.nothing         = 0
    
    return nexus  