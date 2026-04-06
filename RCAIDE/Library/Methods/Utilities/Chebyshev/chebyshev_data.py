# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import RNUMPY as rp

# ----------------------------------------------------------------------
#  Method
# ---------------------------------------------------------------------- 
def chebyshev_data(N = 16, integration = True, **options):
    """Calculates the differentiation and integration matricies
    using chebyshev's pseudospectral algorithm, based on cosine
    spaced samples in x.
    
    D and I are not symmetric
    get derivatives with df_dy = rp.dot(D,f)
    get integral with    int_f = rp.dot(I,f)
        where f is either a 1-d vector or 2-d column array
        
    A full example is available in the function code.

    Assumptions:
    None

    Source:
    N/A

    Inputs:
    N                      [-]        Number of points
    integration (optional) <boolean>  Determines if the integration operator is calculated

    Outputs:
    x                      [-]        N-number of cosine spaced control points, in range [0,1]
    D                      [-]        Differentiation operation matrix
    I                      [-]        Integration operation matrix, or None if integration = False

    Properties Used:
    N/A
    """       
    
    # setup
    N = int(N)
    if N <= 0: raise RuntimeError("N = %i, must be > 0" % N)
    
    
    # --- X vector
    
    # cosine spaced in range [0,1]
    x = 0.5*(1 - rp.cos(rp.pi*rp.arange(0,N)/(N-1)))    


    # --- Differentiation Operator
    
    # coefficients
    c = rp.array( [2.] + [1.]*(N-2) + [2.] )
    c = c * ( (-1.) ** rp.arange(0,N) )
    A = rp.tile( x, (N,1) ).T
    dA = A - A.T + rp.eye( N )
    cinv = 1./c; 

    # build operator
    D = rp.zeros( (N,N) );
    
    # math
    c    = rp.array(c)
    cinv = rp.array([cinv])
    cs   = rp.multiply(c,cinv.T)
    D    = rp.divide(cs.T,dA)

    # more math
    D = D - rp.diag( rp.sum( D.T, axis=0 ) );

    # --- Integration operator
    
    if integration:
        # invert D except first row and column
        I = rp.linalg.inv(D[1:,1:]); 
        
        # repack missing columns with zeros
        I = rp.append(rp.zeros((1,N-1)),I,axis=0)
        I = rp.append(rp.zeros((N,1)),I,axis=1)
        
    else:
        I = None
         
    return x, D, I