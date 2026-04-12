# RCAIDE/Framework/Optimization/Common/helper_functions.py 

# ----------------------------------------------------------------------------------------------------------------- 
#  IMPORT
# --- ------------------------------------------------------------------------------------------------------------- 
  
import RNUMPY as rp

# ----------------------------------------------------------------------------------------------------------------- 
#  Set Values
# -----------------------------------------------------------------------------------------------------------------  
def set_values(dictionary,input_dictionary,converted_values,aliases):
    """ This method regresses through a dictionary to set the required values.
        dictionary is the base class that will be modified, input_dictionary is
        the set of inputs to be used, converted_values are values to be set in the
        base dictionary, and finally the aliases which are where in the dictionary
        the names link to

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    input_dictionary [Data()]
    converted_values [Data()]
    aliases          [list of str]

    Outputs:
    None

    Properties Used:
    N/A
    """      
    
    provided_names = input_dictionary.name
        
    # Correspond aliases to inputs
    pointer = []
    for ii in range(0,len(provided_names)):
        for jj in range(0,len(aliases)):
            if provided_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])

    for ii in range(0,len(pointer)):
        pointers = pointer[ii][:]
        if isinstance(pointers,str):
            if '*' in pointers:
                newstrings = find_a_star(dictionary,pointer[ii])
                for jj in range(0,len(newstrings)):
                    localstrs = newstrings[jj].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])
                    
            elif '*' not in pointers:
                localstrs = pointers.split('.')
                correctname = '.'.join(localstrs[0:])
                dictionary.deep_set(correctname,converted_values[ii])              
        else:
            for zz in range(1,len(pointers)+1):
                if '*' in pointers[zz-1]:
                    newstrings = find_a_star(dictionary,pointer[ii][zz-1])
                    for jj in range(0,len(newstrings)):
                        localstrs = newstrings[jj].split('.')
                        correctname = '.'.join(localstrs[0:])
                        dictionary.deep_set(correctname,converted_values[ii])
                        
                elif '*' not in pointers[zz-1]:
                    localstrs = pointers[zz-1].split('.')
                    correctname = '.'.join(localstrs[0:])
                    dictionary.deep_set(correctname,converted_values[ii])            
            
    return dictionary
        

def find_a_star(dictionary,string):
    """ Searches through a dictionary looking for an *

    Assumptions:
    There may or may not be an asterisk

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    input_dictionary [Data()]
    converted_values [Data()]
    aliases          [list of str]

    Outputs:
    newstrings       [list of str]

    Properties Used:
    N/A
    """
    splitstring = string.split('.')
    for ii in range(0,len(splitstring)):
        if '*' in splitstring[ii]:
            if ii==0:
                newkeys = dictionary.keys()
            elif ii !=0:
                strtoeval = 'dictionary.'+'.'.join(splitstring[0:ii])+'.keys()'
                newkeys = list(eval(strtoeval))
            lastindex   = ii
            
    newstrings = []
    for ii in range(0,len(newkeys)):
        newstrings.append('.'.join(splitstring[0:lastindex])+'.'+newkeys[ii]+'.'+'.'.join(splitstring[lastindex+1:]))
        
    return newstrings


def scale_input_values(inputs,x):
    """ Scales the values according to the a provided scale

    Assumptions:
    

    Source:
    N/A

    Inputs:
    x                [array]         
    inputs           [list]

    Outputs:
    inputs           [list]

    Properties Used:
    N/A
    """    
    
    provided_scale = rp.array(inputs.value[:,-2])
    
    # Safely reconstruct the arrays using hstack
    col_0     = (x * provided_scale).reshape(-1, 1)
    cols_rest = inputs.value[:, 1:]
    
    inputs.value = rp.hstack((col_0, cols_rest))

    return inputs

def limit_input_values(inputs):
    """ Ensures that the inputs are between the bounds

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    x                [array]         
    inputs           [list]

    Outputs:
    inputs           [list]

    Properties Used:
    N/A
    """      
    
    provided_values = inputs[:,1]
    lower_bounds    = inputs[:,2]
    upper_bounds    = inputs[:,3]
    
    # Avoid in-place update for autograd using rp.where and cloning
    provided_values = rp.where(provided_values < lower_bounds, lower_bounds, provided_values)
    provided_values = rp.where(provided_values > upper_bounds, upper_bounds, provided_values)
    
    new_inputs = inputs * 1.0
    new_inputs = new_inputs.at[:, 1].set(provided_values)
    
    return new_inputs
    


def convert_values(inputs): 
    """ Converts an inputs from an optimization into the right units

    Assumptions:
    Always multiply the units by 1!

    Source:
    N/A

    Inputs:
    inputs           [list]

    Outputs:
    converted_values [list of str]

    Properties Used:
    N/A
    """    
    
    provided_values  = inputs.value[:,0]
    
    # Most important 2 lines of these functions
    provided_units   = rp.array(inputs.value[:,-1]*1.0)
    
    # Avoid in-place update for autograd
    new_val = inputs.value * 1.0
    new_val = new_val.at[:,-1].set(provided_units)
    inputs.value = new_val
    
    converted_values = provided_values*provided_units
    
    return converted_values


# ----------------------------------------------------------------------        
#   Get
# ----------------------------------------------------------------------  


def get_values(dictionary,outputs,aliases):
    """ Retrieves values saved in a dictionary 

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    dictionary       [Data()]
    outputs          [Data()]
    aliases          [list of str]

    Outputs:
    values           [float]

    Properties Used:
    N/A
    """     
    try:
        output_names = outputs.name_signs[:,0]
    except:
        output_names = outputs.name[:,0]
        
    # Correspond aliases to outputs
    pointer = []
    for ii in range(0,len(output_names)):
        for jj in range(0,len(aliases)):
            if output_names[ii] == aliases[jj][0]:
                pointer.append(aliases[jj][1])    
                
    values_list = []
    for ii in range(0,len(outputs.value)):
        
        if isinstance(pointer[ii], str) and not ('*' in pointer[ii]) :
            splitstring = pointer[ii].split('.')
            
        else :
            raise TypeError("Pointers for Objectives and Constraints must be unique path (str), not list or contain asterisk")
            
        val  = eval('dictionary.'+'.'.join(splitstring[0:]))
        values_list.append(val)
    
    values = rp.array(values_list)
    
    return values


def scale_obj_values(inputs,x):
    """ Rescales an objective based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs          [Data()]
    x               [float]

    Outputs:
    scaled          [float]

    Properties Used:
    N/A
    """     
    
    # Avoid in-place update for autograd
    provided_scale = inputs.value[:,0]
    provided_units = inputs.value[:,-1]*1.0
    new_val = inputs.value * 1.0
    new_val = new_val.at[:,-1].set(provided_units)
    inputs.value = new_val
    
    scaled =  x/(provided_scale*provided_units)
    
    return scaled


def scale_const_values(inputs,x):
    """ Rescales constraint values based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs          [Data()]
    x               [array]

    Outputs:
    scaled          [array]

    Properties Used:
    N/A
    """        
    
    provided_scale = inputs.value[:,1]
    scaled =  x/provided_scale
    
    return scaled


def scale_const_bnds(inputs):
    """ Rescales constraint bounds based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs           [Data()]

    Outputs:
    converted_values [array]

    Properties Used:
    N/A
    """     
    
    provided_bounds = inputs.value[:,0]
    
    # Avoid in-place update for autograd
    provided_units  = inputs.value[:,-1]*1.0
    new_val = inputs.value * 1.0
    new_val = new_val.at[:,-1].set(provided_units)
    inputs.value = new_val
    
    converted_values = provided_bounds*provided_units
    
    return converted_values


def unscale_const_values(inputs,x):
    """ Rescales values based on Nexus inputs scale

    Assumptions:
    N/A

    Source:
    N/A

    Inputs:
    inputs           [Data()]
    x                [array]

    Outputs:
    scaled           [array]

    Properties Used:
    N/A
    """     
    
    provided_units   = inputs[:,-1]*1.0
    provided_scale = rp.array(inputs[:,-2],dtype = float)
    scaled =  x*provided_scale/provided_units
    
    return scaled
