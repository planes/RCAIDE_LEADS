# generate_vortex_distribution.py
# 
# Created:  May 2018, M. Clarke
# Modified: Apr 2020, M. Clarke
#           Jun 2021, A. Blaufox

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import RCAIDE
from RCAIDE.Framework.Core                                                              import  Data
from RCAIDE.Library.Components.Wings.All_Moving_Surface                                 import All_Moving_Surface 
from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.generate_VD_helpers      import postprocess_VD
from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.make_VLM_wings           import make_VLM_wings 
from RCAIDE.Library.Methods.Aerodynamics.Vortex_Lattice_Method.deflect_control_surface  import deflect_control_surface
from RCAIDE.Library.Methods.Geometry.Airfoil                                            import compute_naca_4series, import_airfoil_geometry
  
# package imports 
import RNUMPY as rp

# ----------------------------------------------------------------------
#  Generate Vortex Distribution
# ----------------------------------------------------------------------
def generate_vortex_distribution(conditions,settings,geometry): 
    precision      = settings.floating_point_precision    
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD_seg = Data()

    VD_seg.XAH    = rp.empty(shape=[0,0], dtype=precision)
    VD_seg.YAH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZAH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.XBH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YBH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZBH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.XCH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YCH    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZCH    = rp.empty(shape=[0,1], dtype=precision)     
    VD_seg.XA1    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YA1    = rp.empty(shape=[0,1], dtype=precision)  
    VD_seg.ZA1    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.XA2    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YA2    = rp.empty(shape=[0,1], dtype=precision)    
    VD_seg.ZA2    = rp.empty(shape=[0,1], dtype=precision)    
    VD_seg.XB1    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YB1    = rp.empty(shape=[0,1], dtype=precision)  
    VD_seg.ZB1    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.XB2    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YB2    = rp.empty(shape=[0,1], dtype=precision)    
    VD_seg.ZB2    = rp.empty(shape=[0,1], dtype=precision)     
    VD_seg.XAC    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YAC    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZAC    = rp.empty(shape=[0,1], dtype=precision) 
    VD_seg.XBC    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YBC    = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZBC    = rp.empty(shape=[0,1], dtype=precision) 
    VD_seg.XC_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YC_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZC_TE  = rp.empty(shape=[0,1], dtype=precision)     
    VD_seg.XA_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YA_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZA_TE  = rp.empty(shape=[0,1], dtype=precision) 
    VD_seg.XB_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YB_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZB_TE  = rp.empty(shape=[0,1], dtype=precision)  
    VD_seg.XC     = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.YC     = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.ZC     = rp.empty(shape=[0,1], dtype=precision)        
    VD_seg.CS     = rp.empty(shape=[0,1], dtype=precision) 
    VD_seg.X      = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.Y      = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.Z      = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.Y_SW   = rp.empty(shape=[0,1], dtype=precision)
    VD_seg.DY     = rp.empty(shape=[0,1], dtype=precision)
     
    for i in range(len(conditions.aerodynamics.angles.alpha)):
        VD_i = generate_control_point_vortex_distribution(geometry,settings)  
        if i ==  0: 
            VD_seg.XAH    = rp.atleast_2d(VD_i.XAH   ) 
            VD_seg.YAH    = rp.atleast_2d(VD_i.YAH   )
            VD_seg.ZAH    = rp.atleast_2d(VD_i.ZAH   )
            VD_seg.XBH    = rp.atleast_2d(VD_i.XBH   )
            VD_seg.YBH    = rp.atleast_2d(VD_i.YBH   )
            VD_seg.ZBH    = rp.atleast_2d(VD_i.ZBH   )
            VD_seg.XCH    = rp.atleast_2d(VD_i.XCH   )
            VD_seg.YCH    = rp.atleast_2d(VD_i.YCH   )
            VD_seg.ZCH    = rp.atleast_2d(VD_i.ZCH   )     
            VD_seg.XA1    = rp.atleast_2d(VD_i.XA1   )
            VD_seg.YA1    = rp.atleast_2d(VD_i.YA1   )  
            VD_seg.ZA1    = rp.atleast_2d(VD_i.ZA1   )
            VD_seg.XA2    = rp.atleast_2d(VD_i.XA2   )
            VD_seg.YA2    = rp.atleast_2d(VD_i.YA2   )    
            VD_seg.ZA2    = rp.atleast_2d(VD_i.ZA2   )    
            VD_seg.XB1    = rp.atleast_2d(VD_i.XB1   )
            VD_seg.YB1    = rp.atleast_2d(VD_i.YB1   )  
            VD_seg.ZB1    = rp.atleast_2d(VD_i.ZB1   )
            VD_seg.XB2    = rp.atleast_2d(VD_i.XB2   )
            VD_seg.YB2    = rp.atleast_2d(VD_i.YB2   )    
            VD_seg.ZB2    = rp.atleast_2d(VD_i.ZB2   )     
            VD_seg.XAC    = rp.atleast_2d(VD_i.XAC   )
            VD_seg.YAC    = rp.atleast_2d(VD_i.YAC   )
            VD_seg.ZAC    = rp.atleast_2d(VD_i.ZAC   ) 
            VD_seg.XBC    = rp.atleast_2d(VD_i.XBC   )
            VD_seg.YBC    = rp.atleast_2d(VD_i.YBC   )
            VD_seg.ZBC    = rp.atleast_2d(VD_i.ZBC   ) 
            VD_seg.XC_TE  = rp.atleast_2d(VD_i.XC_TE )
            VD_seg.YC_TE  = rp.atleast_2d(VD_i.YC_TE )
            VD_seg.ZC_TE  = rp.atleast_2d(VD_i.ZC_TE )     
            VD_seg.XA_TE  = rp.atleast_2d(VD_i.XA_TE )
            VD_seg.YA_TE  = rp.atleast_2d(VD_i.YA_TE )
            VD_seg.ZA_TE  = rp.atleast_2d(VD_i.ZA_TE ) 
            VD_seg.XB_TE  = rp.atleast_2d(VD_i.XB_TE )
            VD_seg.YB_TE  = rp.atleast_2d(VD_i.YB_TE )
            VD_seg.ZB_TE  = rp.atleast_2d(VD_i.ZB_TE )  
            VD_seg.XC     = rp.atleast_2d(VD_i.XC    )
            VD_seg.YC     = rp.atleast_2d(VD_i.YC    )
            VD_seg.ZC     = rp.atleast_2d(VD_i.ZC    )         
            VD_seg.CS     = rp.atleast_2d(VD_i.CS    ) 
            VD_seg.X      = rp.atleast_2d(VD_i.X     )
            VD_seg.Y      = rp.atleast_2d(VD_i.Y     )
            VD_seg.Z      = rp.atleast_2d(VD_i.Z     )
            VD_seg.Y_SW   = rp.atleast_2d(VD_i.Y_SW  )
            VD_seg.DY     = rp.atleast_2d(VD_i.DY    )  
            VD_seg.n_w                       =  rp.atleast_2d(VD_i.n_w                      )
            VD_seg.n_cp                      =  rp.atleast_2d(VD_i.n_cp                     )
            VD_seg.n_sw                      =  rp.atleast_2d(VD_i.n_sw                     )
            VD_seg.n_cw                      =  rp.atleast_2d(VD_i.n_cw                     )
            VD_seg.chordwise_breaks          =  rp.atleast_2d(VD_i.chordwise_breaks         )
            VD_seg.spanwise_breaks           =  rp.atleast_2d(VD_i.spanwise_breaks          )
            VD_seg.symmetric_wings           =  rp.atleast_2d(VD_i.symmetric_wings          )
            VD_seg.surface_ID                =  rp.atleast_2d(VD_i.surface_ID               )
            VD_seg.surface_ID_full           =  rp.atleast_2d(VD_i.surface_ID_full          )
            VD_seg.leading_edge_indices      =  rp.atleast_2d(VD_i.leading_edge_indices     )
            VD_seg.leading_edge_sweeps       =  rp.atleast_2d(VD_i.leading_edge_sweeps      )
            VD_seg.trailing_edge_indices     =  rp.atleast_2d(VD_i.trailing_edge_indices    )
            VD_seg.panels_per_strip          =  rp.atleast_2d(VD_i.panels_per_strip         )
            VD_seg.chordwise_panel_number    =  rp.atleast_2d(VD_i.chordwise_panel_number   )
            VD_seg.chord_lengths             =  rp.atleast_2d(VD_i.chord_lengths            )
            VD_seg.chord_widths              =  rp.atleast_2d(VD_i.chord_widths            )
            VD_seg.tangent_incidence_angle   =  rp.atleast_2d(VD_i.tangent_incidence_angle  )
            VD_seg.exposed_leading_edge_flag =  rp.atleast_2d(VD_i.exposed_leading_edge_flag) 
            VD_seg.wing_areas                =  rp.atleast_2d(VD_i.wing_areas               )
            VD_seg.vortex_lift               =  rp.atleast_2d(VD_i.vortex_lift              )
            VD_seg.counter                   =  rp.atleast_2d(VD_i.counter                  ) 
            VD_seg.panel_areas               =  rp.atleast_2d(VD_i.panel_areas              )
            VD_seg.normals                   =  VD_i.normals[None,:, :]
            VD_seg.SLOPE                     =  rp.atleast_2d(VD_i.SLOPE                    )
            VD_seg.SLE                       =  rp.atleast_2d(VD_i.SLE                      )
            VD_seg.D                         =  rp.atleast_2d(VD_i.D                        )
            VD_seg.tangent_incidence_angle   =  rp.atleast_2d(VD_i.tangent_incidence_angle  ) 
            VD_seg.VLM_wings                 =  VD_i.VLM_wings
            VD_seg.is_postprocessed          =  VD_i.is_postprocessed
            
        else:

            VD_seg.XAH    = rp.vstack(( VD_seg.XAH  , rp.atleast_2d(VD_i.XAH   )))
            VD_seg.YAH    = rp.vstack(( VD_seg.YAH  , rp.atleast_2d(VD_i.YAH   )))
            VD_seg.ZAH    = rp.vstack(( VD_seg.ZAH  , rp.atleast_2d(VD_i.ZAH   )))
            VD_seg.XBH    = rp.vstack(( VD_seg.XBH  , rp.atleast_2d(VD_i.XBH   )))
            VD_seg.YBH    = rp.vstack(( VD_seg.YBH  , rp.atleast_2d(VD_i.YBH   )))
            VD_seg.ZBH    = rp.vstack(( VD_seg.ZBH  , rp.atleast_2d(VD_i.ZBH   )))
            VD_seg.XCH    = rp.vstack(( VD_seg.XCH  , rp.atleast_2d(VD_i.XCH   )))
            VD_seg.YCH    = rp.vstack(( VD_seg.YCH  , rp.atleast_2d(VD_i.YCH   )))
            VD_seg.ZCH    = rp.vstack(( VD_seg.ZCH  , rp.atleast_2d(VD_i.ZCH   )))     
            VD_seg.XA1    = rp.vstack(( VD_seg.XA1  , rp.atleast_2d(VD_i.XA1   )))
            VD_seg.YA1    = rp.vstack(( VD_seg.YA1  , rp.atleast_2d(VD_i.YA1   )))  
            VD_seg.ZA1    = rp.vstack(( VD_seg.ZA1  , rp.atleast_2d(VD_i.ZA1   )))
            VD_seg.XA2    = rp.vstack(( VD_seg.XA2  , rp.atleast_2d(VD_i.XA2   )))
            VD_seg.YA2    = rp.vstack(( VD_seg.YA2  , rp.atleast_2d(VD_i.YA2   )))    
            VD_seg.ZA2    = rp.vstack(( VD_seg.ZA2  , rp.atleast_2d(VD_i.ZA2   )))    
            VD_seg.XB1    = rp.vstack(( VD_seg.XB1  , rp.atleast_2d(VD_i.XB1   )))
            VD_seg.YB1    = rp.vstack(( VD_seg.YB1  , rp.atleast_2d(VD_i.YB1   )))  
            VD_seg.ZB1    = rp.vstack(( VD_seg.ZB1  , rp.atleast_2d(VD_i.ZB1   )))
            VD_seg.XB2    = rp.vstack(( VD_seg.XB2  , rp.atleast_2d(VD_i.XB2   )))
            VD_seg.YB2    = rp.vstack(( VD_seg.YB2  , rp.atleast_2d(VD_i.YB2   )))    
            VD_seg.ZB2    = rp.vstack(( VD_seg.ZB2  , rp.atleast_2d(VD_i.ZB2   )))     
            VD_seg.XAC    = rp.vstack(( VD_seg.XAC  , rp.atleast_2d(VD_i.XAC   )))
            VD_seg.YAC    = rp.vstack(( VD_seg.YAC  , rp.atleast_2d(VD_i.YAC   )))
            VD_seg.ZAC    = rp.vstack(( VD_seg.ZAC  , rp.atleast_2d(VD_i.ZAC   ))) 
            VD_seg.XBC    = rp.vstack(( VD_seg.XBC  , rp.atleast_2d(VD_i.XBC   )))
            VD_seg.YBC    = rp.vstack(( VD_seg.YBC  , rp.atleast_2d(VD_i.YBC   )))
            VD_seg.ZBC    = rp.vstack(( VD_seg.ZBC  , rp.atleast_2d(VD_i.ZBC   ))) 
            VD_seg.XC_TE  = rp.vstack(( VD_seg.XC_TE, rp.atleast_2d(VD_i.XC_TE )))
            VD_seg.YC_TE  = rp.vstack(( VD_seg.YC_TE, rp.atleast_2d(VD_i.YC_TE )))
            VD_seg.ZC_TE  = rp.vstack(( VD_seg.ZC_TE, rp.atleast_2d(VD_i.ZC_TE )))     
            VD_seg.XA_TE  = rp.vstack(( VD_seg.XA_TE, rp.atleast_2d(VD_i.XA_TE )))
            VD_seg.YA_TE  = rp.vstack(( VD_seg.YA_TE, rp.atleast_2d(VD_i.YA_TE )))
            VD_seg.ZA_TE  = rp.vstack(( VD_seg.ZA_TE, rp.atleast_2d(VD_i.ZA_TE ))) 
            VD_seg.XB_TE  = rp.vstack(( VD_seg.XB_TE, rp.atleast_2d(VD_i.XB_TE )))
            VD_seg.YB_TE  = rp.vstack(( VD_seg.YB_TE, rp.atleast_2d(VD_i.YB_TE )))
            VD_seg.ZB_TE  = rp.vstack(( VD_seg.ZB_TE, rp.atleast_2d(VD_i.ZB_TE )))  
            VD_seg.XC     = rp.vstack(( VD_seg.XC   , rp.atleast_2d(VD_i.XC    )))
            VD_seg.YC     = rp.vstack(( VD_seg.YC   , rp.atleast_2d(VD_i.YC    )))
            VD_seg.ZC     = rp.vstack(( VD_seg.ZC   , rp.atleast_2d(VD_i.ZC    )))       
            VD_seg.CS     = rp.vstack(( VD_seg.CS   , rp.atleast_2d(VD_i.CS    ))) 
            VD_seg.X      = rp.vstack(( VD_seg.X    , rp.atleast_2d(VD_i.X     )))
            VD_seg.Y      = rp.vstack(( VD_seg.Y    , rp.atleast_2d(VD_i.Y     )))
            VD_seg.Z      = rp.vstack(( VD_seg.Z    , rp.atleast_2d(VD_i.Z     )))
            VD_seg.Y_SW   = rp.vstack(( VD_seg.Y_SW , rp.atleast_2d(VD_i.Y_SW  )))
            VD_seg.DY     = rp.vstack(( VD_seg.DY   , rp.atleast_2d(VD_i.DY    ))) 

            VD_seg.n_w                       = rp.vstack((VD_seg.n_w                        , rp.atleast_2d(VD_i.n_w                      )))
            VD_seg.n_cp                      = rp.vstack((VD_seg.n_cp                       , rp.atleast_2d(VD_i.n_cp                     )))
            VD_seg.n_sw                      = rp.vstack((VD_seg.n_sw                       , rp.atleast_2d(VD_i.n_sw                     )))
            VD_seg.n_cw                      = rp.vstack((VD_seg.n_cw                       , rp.atleast_2d(VD_i.n_cw                     )))
            VD_seg.chordwise_breaks          = rp.vstack((VD_seg.chordwise_breaks           , rp.atleast_2d(VD_i.chordwise_breaks         )))
            VD_seg.spanwise_breaks           = rp.vstack((VD_seg.spanwise_breaks            , rp.atleast_2d(VD_i.spanwise_breaks          )))
            VD_seg.symmetric_wings           = rp.vstack((VD_seg.symmetric_wings            , rp.atleast_2d(VD_i.symmetric_wings          )))
            VD_seg.surface_ID                = rp.vstack((VD_seg.surface_ID                 , rp.atleast_2d(VD_i.surface_ID               )))
            VD_seg.surface_ID_full           = rp.vstack((VD_seg.surface_ID_full            , rp.atleast_2d(VD_i.surface_ID_full          )))
            VD_seg.leading_edge_indices      = rp.vstack((VD_seg.leading_edge_indices       , rp.atleast_2d(VD_i.leading_edge_indices     )))
            VD_seg.leading_edge_sweeps       = rp.vstack((VD_seg.leading_edge_sweeps        , rp.atleast_2d(VD_i.leading_edge_sweeps     )))
            VD_seg.trailing_edge_indices     = rp.vstack((VD_seg.trailing_edge_indices      , rp.atleast_2d(VD_i.trailing_edge_indices    )))
            VD_seg.panels_per_strip          = rp.vstack((VD_seg.panels_per_strip           , rp.atleast_2d(VD_i.panels_per_strip         )))
            VD_seg.chordwise_panel_number    = rp.vstack((VD_seg.chordwise_panel_number     , rp.atleast_2d(VD_i.chordwise_panel_number   )))
            VD_seg.chord_lengths             = rp.vstack((VD_seg.chord_lengths              , rp.atleast_2d(VD_i.chord_lengths            )))
            VD_seg.chord_widths              = rp.vstack((VD_seg.chord_widths              , rp.atleast_2d(VD_i.chord_widths              )))
            VD_seg.tangent_incidence_angle   = rp.vstack((VD_seg.tangent_incidence_angle    , rp.atleast_2d(VD_i.tangent_incidence_angle  )))
            VD_seg.exposed_leading_edge_flag = rp.vstack((VD_seg.exposed_leading_edge_flag  , rp.atleast_2d(VD_i.exposed_leading_edge_flag)))
            VD_seg.wing_areas                = rp.vstack((VD_seg.wing_areas                 , rp.atleast_2d(VD_i.wing_areas               )))
            VD_seg.vortex_lift               = rp.vstack((VD_seg.vortex_lift                , rp.atleast_2d(VD_i.vortex_lift              )))
            VD_seg.counter                   = rp.vstack((VD_seg.counter                    , rp.atleast_2d(VD_i.counter                  ))) 
            VD_seg.panel_areas               = rp.vstack((VD_seg.panel_areas                , rp.atleast_2d(VD_i.panel_areas              )))
            VD_seg.normals                   = rp.vstack((VD_seg.normals                    , VD_i.normals[None, :, :]                    ))
            VD_seg.SLOPE                     = rp.vstack((VD_seg.SLOPE                      , rp.atleast_2d(VD_i.SLOPE                    )))
            VD_seg.SLE                       = rp.vstack((VD_seg.SLE                        , rp.atleast_2d(VD_i.SLE                      )))
            VD_seg.D                         = rp.vstack((VD_seg.D                          , rp.atleast_2d(VD_i.D                        )))   
    
    return VD_seg 
    
    
def generate_control_point_vortex_distribution(geometry,settings):
    ''' Compute the coordinates of panels, vortices , control points
    and geometry used to build the influence coefficient matrix. A 
    different discretization (n_sw and n_cw) may be defined for each type
    of major section (wings and fuselages). 
    
    Control surfaces are modelled as wings, but adapt their panel density 
    to that of the area in which they reside on their own wing.   

    Assumptions: 
    Below is a schematic of the coordinates of an arbitrary panel  
    
    XA1 ____________________________ XB1    
       |                            |
       |        bound vortex        |
    XAH|  ________________________  |XBH
       | |           XCH          | |
       | |                        | |
       | |                        | |     
       | |                        | |
       | |                        | |
       | |           0 <--control | |       
       | |          XC     point  | |  
       | |                        | |
   XA2 |_|________________________|_|XB2
         |                        |     
         |       trailing         |  
         |   <--  vortex   -->    |  
         |         legs           | 
             
    
    In addition, all control surfaces should be appended directly
       to the wing, not the wing segments    
    
    For control surfaces, "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    
    Source:  
    None

    Inputs:
    geometry.wings                                [Unitless]  
    settings.floating_point_precision             [rp.dtype]
    
    Of the following settings, the user should define either the number_ atrributes or the wing_ and fuse_ attributes.
    settings.number_of_spanwise_vortices             - a base number of vortices to be applied to both wings and fuselages
    settings.number_of_chordwise_vortices            - a base number of vortices to be applied to both wings and fuselages
    settings.wing_spanwise_vortices               - the number of vortices to be applied to only the wings
    settings.wing_chordwise_vortices              - the number of vortices to be applied to only the wings
    settings.fuselage_spanwise_vortices           - the number of vortices to be applied to only the fuslages
    settings.fuselage_chordwise_vortices          - the number of vortices to be applied to only the fuselages 
       
    Outputs:                                   
    VD - vehicle vortex distribution              [Unitless] 

    Properties Used:
    N/A 
         
    '''
    # ---------------------------------------------------------------------------------------
    # STEP 0: Unpack settings
    # ---------------------------------------------------------------------------------------        
    #unpack other settings----------------------------------------------------
    spc            = settings.spanwise_cosine_spacing
    model_fuselage = settings.model_fuselage 
    precision      = settings.floating_point_precision
    
    show_prints    = settings.verbose if ('verbose' in settings.keys()) else False
    
    # unpack discretization settings------------------------------------------
    n_sw_global    = settings.number_of_spanwise_vortices
    n_cw_global    = settings.number_of_chordwise_vortices
    n_sw_wing      = settings.wing_spanwise_vortices  
    n_cw_wing      = settings.wing_chordwise_vortices
    n_sw_fuse      = settings.fuselage_spanwise_vortices  
    n_cw_fuse      = settings.fuselage_chordwise_vortices
    
    #make sure n_cw and n_sw are both defined or not defined
    invalid_global_n   = bool(n_sw_global) != bool(n_cw_global) 
    invalid_wing_n     = bool(n_sw_wing)   != bool(n_cw_wing)
    invalid_fuse_n     = bool(n_sw_fuse)   != bool(n_cw_fuse)
    invalid_separate_n = bool(n_sw_wing)   != bool(n_sw_fuse)
    if invalid_global_n:
        raise AssertionError('If using global surface discretization, both n_sw and n_cw must be defined')
    elif invalid_wing_n or invalid_fuse_n:
        raise AssertionError('If using separate surface discretization, all n_sw and n_cw values must be defined')
    elif invalid_separate_n:
        raise AssertionError('If using separate surface discretization, both wing and fuselage discretization must be defined')
    
    #make sure that global and separate settings aren't both defined
    global_n_defined   = bool(n_sw_global) 
    separate_n_defined = bool(n_sw_wing)
    if global_n_defined == separate_n_defined:
        raise AssertionError('Specify either global or separate discretization')
    elif global_n_defined:
        n_sw_wing = n_sw_global
        n_cw_wing = n_cw_global
        n_sw_fuse = n_sw_global
        n_cw_fuse = n_cw_global
    else: #separate_n_defined
        #everything is already set up to use separate discretization
        pass
    
    # ---------------------------------------------------------------------------------------
    # STEP 1: Define empty vectors for coordinates of panes, control points and bound vortices
    # ---------------------------------------------------------------------------------------
    VD = Data()

    VD.XAH    = rp.empty(shape=[0,1], dtype=precision)
    VD.YAH    = rp.empty(shape=[0,1], dtype=precision)
    VD.ZAH    = rp.empty(shape=[0,1], dtype=precision)
    VD.XBH    = rp.empty(shape=[0,1], dtype=precision)
    VD.YBH    = rp.empty(shape=[0,1], dtype=precision)
    VD.ZBH    = rp.empty(shape=[0,1], dtype=precision)
    VD.XCH    = rp.empty(shape=[0,1], dtype=precision)
    VD.YCH    = rp.empty(shape=[0,1], dtype=precision)
    VD.ZCH    = rp.empty(shape=[0,1], dtype=precision)     
    VD.XA1    = rp.empty(shape=[0,1], dtype=precision)
    VD.YA1    = rp.empty(shape=[0,1], dtype=precision)  
    VD.ZA1    = rp.empty(shape=[0,1], dtype=precision)
    VD.XA2    = rp.empty(shape=[0,1], dtype=precision)
    VD.YA2    = rp.empty(shape=[0,1], dtype=precision)    
    VD.ZA2    = rp.empty(shape=[0,1], dtype=precision)    
    VD.XB1    = rp.empty(shape=[0,1], dtype=precision)
    VD.YB1    = rp.empty(shape=[0,1], dtype=precision)  
    VD.ZB1    = rp.empty(shape=[0,1], dtype=precision)
    VD.XB2    = rp.empty(shape=[0,1], dtype=precision)
    VD.YB2    = rp.empty(shape=[0,1], dtype=precision)    
    VD.ZB2    = rp.empty(shape=[0,1], dtype=precision)     
    VD.XAC    = rp.empty(shape=[0,1], dtype=precision)
    VD.YAC    = rp.empty(shape=[0,1], dtype=precision)
    VD.ZAC    = rp.empty(shape=[0,1], dtype=precision) 
    VD.XBC    = rp.empty(shape=[0,1], dtype=precision)
    VD.YBC    = rp.empty(shape=[0,1], dtype=precision)
    VD.ZBC    = rp.empty(shape=[0,1], dtype=precision) 
    VD.XC_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.YC_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.ZC_TE  = rp.empty(shape=[0,1], dtype=precision)     
    VD.XA_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.YA_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.ZA_TE  = rp.empty(shape=[0,1], dtype=precision) 
    VD.XB_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.YB_TE  = rp.empty(shape=[0,1], dtype=precision)
    VD.ZB_TE  = rp.empty(shape=[0,1], dtype=precision)  
    VD.XC     = rp.empty(shape=[0,1], dtype=precision)
    VD.YC     = rp.empty(shape=[0,1], dtype=precision)
    VD.ZC     = rp.empty(shape=[0,1], dtype=precision)         
    VD.CS     = rp.empty(shape=[0,1], dtype=precision) 
    VD.X      = rp.empty(shape=[0,1], dtype=precision)
    VD.Y      = rp.empty(shape=[0,1], dtype=precision)
    VD.Z      = rp.empty(shape=[0,1], dtype=precision)
    VD.Y_SW   = rp.empty(shape=[0,1], dtype=precision)
    VD.DY     = rp.empty(shape=[0,1], dtype=precision) 

    # empty vectors necessary for arbitrary discretization dimensions
    VD.n_w              = 0                            # number of wings counter (refers to wings, fuselages or other structures)  
    VD.n_cp             = 0                            # number of bound vortices (panels) counter 
    VD.n_sw             = rp.array([], dtype=rp.int16) # array of the number of spanwise  strips in each wing
    VD.n_cw             = rp.array([], dtype=rp.int16) # array of the number of chordwise panels per strip in each wing
    VD.chordwise_breaks = rp.array([], dtype=rp.int32) # indices of the first panel in every strip      (given a list of all panels)
    VD.spanwise_breaks  = rp.array([], dtype=rp.int32) # indices of the first strip of panels in a wing (given chordwise_breaks)    
    VD.symmetric_wings  = rp.array([], dtype=rp.int32)
    VD.surface_ID       = rp.empty(shape=[0,1], dtype=rp.int16) 
    VD.surface_ID_full  = rp.empty(shape=[0,1], dtype=rp.int16)   
    VD.leading_edge_indices      = rp.array([], dtype=bool)      # bool array of leading  edge indices (all false except for panels at leading  edge)
    VD.leading_edge_sweeps       = rp.array([], dtype=bool)      # bool array of leading  edge indices (all false except for panels at leading  edge)
    VD.trailing_edge_indices     = rp.array([], dtype=bool)      # bool array of trailing edge indices (all false except for panels at trailing edge)    
    VD.panels_per_strip          = rp.array([], dtype=rp.int)  # array of the number of panels per strip (RNMAX); this is assigned for all panels  
    VD.chordwise_panel_number    = rp.array([], dtype=rp.int16)  # array of panels' numbers in their strips.     
    VD.chord_lengths             = rp.array([], dtype=precision) # Chord length, this is assigned for all panels.
    VD.tangent_incidence_angle   = rp.array([], dtype=precision) # Tangent Incidence Angles of the chordwise strip. LE to TE, ZETA
    VD.chord_widths              = rp.array([], dtype=precision) # Chord width, this is assigned for all panels.
    VD.exposed_leading_edge_flag = rp.array([], dtype=rp.int16)  # 0 or 1 per strip. 0 turns off leading edge suction for non-slat control surfaces
    
    # ---------------------------------------------------------------------------------------
    # Unpack aircraft wing geometry 
    # ---------------------------------------------------------------------------------------    
    VD.wing_areas  = [] # instantiate wing areas
    VD.vortex_lift = []
    VD.counter     = 0
    
    #reformat/preprocess wings and control surfaces for VLM panelization
    VLM_wings    = make_VLM_wings(geometry, settings)
    VD.VLM_wings = VLM_wings
    
    #generate panelization for each wing. Wings first, then control surface wings
    for wing in VD.VLM_wings:
        if not wing.is_a_control_surface:
            if show_prints: print('discretizing ' + wing.tag) 
            VD, wing = generate_wing_vortex_distribution(VD,wing,n_cw_wing,n_sw_wing,spc,precision)    
                    
    for wing in VD.VLM_wings:
        if wing.is_a_control_surface:
            if show_prints:print('discretizing ' + wing.tag)
            VD, wing = generate_wing_vortex_distribution(VD,wing,n_cw_wing,n_sw_wing,spc,precision)     
            
            
    # ---------------------------------------------------------------------------------------
    # Unpack aircraft fuselage geometry
    # ---------------------------------------------------------------------------------------      
    VD.wing_areas = rp.array(VD.wing_areas, dtype=precision)
    VD.n_fus      = 0   
    for fus in geometry.fuselages:
        if show_prints: print('discretizing ' + fus.tag)
        VD = generate_fuselage_and_nacelle_vortex_distribution(VD,fus,n_cw_fuse,n_sw_fuse,precision,model_fuselage)


    # ---------------------------------------------------------------------------------------
    # Deflect Control Surfaces
    # ---------------------------------------------------------------------------------------      
    for wing in VD.VLM_wings:
        wing_is_all_moving = (not wing.is_a_control_surface) and issubclass(wing.wing_type, All_Moving_Surface)        
        if wing.is_a_control_surface or wing_is_all_moving:
            # Deflect the control surface
            VD, wing = deflect_control_surface(VD, wing)
            
    # ---------------------------------------------------------------------------------------
    # Postprocess VD information
    # ---------------------------------------------------------------------------------------   
    VD = postprocess_VD(VD, settings) 
    
    return VD 


# ----------------------------------------------------------------------
#  Discretize Wings
# ----------------------------------------------------------------------
def generate_wing_vortex_distribution(VD,wing,n_cw,n_sw,spc,precision):
    """ This generates vortex distribution points for the given wing 

    Assumptions: 
    The wing is segmented and was made or modified by make_VLM_wings()
    
    For control surfaces, "positve" deflection corresponds to the RH rule where the axis of rotation is the OUTBOARD-pointing hinge vector
    symmetry: the LH rule is applied to the reflected surface for non-ailerons. Ailerons follow a RH rule for both sides
    
    The hinge_vector will only ever be calcualted on the first strip of any control/all-moving surface. It is assumed that all control
    surfaces are trapezoids, thus needing only one hinge, and that all all-moving surfaces have exactly one point of rotation.

    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution
    wing                 - a Data object made or modified by make_VLM_wings() to mimick a Wing object
    
    Properties Used:
    N/A
    """ 
    wings = VD.VLM_wings
    
    # get geometry of wing  
    span          = wing.spans.projected
    sym_para      = wing.xz_plane_symmetric 
    vertical_wing = wing.vertical
    wing_origin   = wing.origin[0]
    VD.vortex_lift.append(wing.vortex_lift)

    # determine if vehicle has symmetry 
    if sym_para is True :
        span = span/2
        VD.vortex_lift.append(wing.vortex_lift)
        
    VD.counter  = VD.counter + 1
    wing.surface_ID = VD.counter*1

    # ---------------------------------------------------------------------------------------
    # STEP 3: Get discretization control variables  
    # ---------------------------------------------------------------------------------------
    # get number of spanwise and chordwise panels for this wing
    n_sw = n_sw if (not wing.is_a_control_surface) else max(len(wing.y_coords_required)-1,1)
    n_cw = n_cw if (not wing.is_a_control_surface) else max(int(rp.ceil(wing.chord_fraction*n_cw)),2)  
    
    # get y_coordinates (y-locations of the edges of each strip in wing-local coords)
    if spc == True: # discretize wing using cosine spacing     
        n               = rp.linspace(n_sw+1,0,n_sw+1)         # vectorize
        thetan          = n*(rp.pi/2)/(n_sw+1)                 # angular stations
        y_coordinates   = span*rp.cos(thetan)                  # y locations based on the angular spacing
    else:           # discretize wing using linear spacing 
        y_coordinates   = rp.linspace(0,span,n_sw+1) 

    # get span_breaks object
    span_breaks   = wing.span_breaks
    n_breaks      = len(span_breaks)

    # ---------------------------------------------------------------------------------------
    # STEP 4: Setup span_break and section arrays. A 'section' is the trapezoid between two
    #         span_breaks. A 'span_break' is described in the file make_VLM_wings.py
    # ---------------------------------------------------------------------------------------
    break_chord       = rp.zeros(n_breaks)
    break_twist       = rp.zeros(n_breaks)
    break_sweep       = rp.zeros(n_breaks)
    break_dihedral    = rp.zeros(n_breaks)
    break_camber_xs   = [] 
    break_camber_zs   = []
    break_x_offset    = rp.zeros(n_breaks)
    break_z_offset    = rp.zeros(n_breaks)
    break_spans       = rp.zeros(n_breaks) 
    section_span      = rp.zeros(n_breaks)
    section_area      = rp.zeros(n_breaks)
    section_LE_cut    = rp.zeros(n_breaks)
    section_TE_cut    = rp.ones(n_breaks)

    # ---------------------------------------------------------------------------------------
    # STEP 5:  Obtain sweep, chord, dihedral and twist at the beginning/end of each break.
    #          If applicable, append airfoil section VD and flap/aileron deflection angles.
    # --------------------------------------------------------------------------------------- 
    for i_break in range(n_breaks):   
        break_spans = break_spans.at[i_break].set(span_breaks[i_break].span_fraction*span)  
        break_chord = break_chord.at[i_break].set(span_breaks[i_break].local_chord)
        break_twist = break_twist.at[i_break].set(span_breaks[i_break].twist)
        break_dihedral = break_dihedral.at[i_break].set(span_breaks[i_break].dihedral_outboard)             

        # get leading edge sweep. make_VLM wings should have precomputed this for all span_breaks
        is_not_last_break    = (i_break != n_breaks-1)
        break_sweep = break_sweep.at[i_break].set(span_breaks[i_break].sweep_outboard_LE if is_not_last_break else 0)

        # find span and area. All span_break offsets should be calculated in make_VLM_wings
        if i_break == 0:
            section_span = section_span.at[i_break].set(0.0)
            break_x_offset = break_x_offset.at[i_break].set(0.0)  
            break_z_offset = break_z_offset.at[i_break].set(0.0)       
        else:
            section_span = section_span.at[i_break].set(break_spans[i_break] - break_spans[i_break-1])
            new_area = 0.5 * (break_chord[i_break-1] + break_chord[i_break]) * section_span[i_break]
            section_area = section_area.at[i_break].set(new_area)
            
            break_x_offset = break_x_offset.at[i_break].set(span_breaks[i_break].x_offset)
            break_z_offset = break_z_offset.at[i_break].set(span_breaks[i_break].dih_offset)

        # Get airfoil section VD  
        if span_breaks[i_break].airfoil: 
            if type(span_breaks[i_break].airfoil) == RCAIDE.Library.Components.Airfoils.NACA_4_Series_Airfoil: # check if naca 4 series of airfoil from datafile
                airfoil_geo_data = compute_naca_4series(span_breaks[i_break].airfoil.NACA_4_Series_code,span_breaks[i_break].airfoil.number_of_points-2)
            else:
                airfoil_geo_data = import_airfoil_geometry(span_breaks[i_break].airfoil.coordinate_file)  
            break_camber_zs.append(airfoil_geo_data.camber_coordinates)
            break_camber_xs.append(airfoil_geo_data.x_lower_surface) 
        else:
            break_camber_zs.append(rp.zeros(30))              
            break_camber_xs.append(rp.linspace(0,1,30)) 

        # Get control surface leading and trailing edge cute cuts: section__cuts[-1] should never be used in the following code
        section_LE_cut = section_LE_cut.at[i_break].set(span_breaks[i_break].cuts[0,1])
        section_TE_cut = section_TE_cut.at[i_break].set(span_breaks[i_break].cuts[1,1])

    VD.wing_areas.append(rp.sum(section_area[:], dtype=precision))
    if sym_para is True :
        VD.wing_areas.append(rp.sum(section_area[:], dtype=precision))            

    #Shift spanwise vortices onto section breaks  
    if len(y_coordinates) < n_breaks:
        raise ValueError('Not enough spanwise VLM stations for segment breaks')

    y_coords_required = break_spans if (not wing.is_a_control_surface) else rp.array(sorted(wing.y_coords_required))  #control surfaces have additional required y_coords  
    shifted_idxs = rp.zeros(len(y_coordinates))
    for y_req in y_coords_required:
        idx = (rp.abs(y_coordinates - y_req) + shifted_idxs).argmin() #index of y-coord nearest to the span break
        shifted_idxs = shifted_idxs.at[idx].set(rp.inf) 
        y_coordinates = y_coordinates.at[idx].set(y_req)

    y_coordinates = rp.array(sorted(y_coordinates))
    
    for y_req in y_coords_required:
        if y_req not in y_coordinates:
            raise ValueError('VLM did not capture all section breaks')  
    
    # ---------------------------------------------------------------------------------------
    # STEP 6: Define coordinates of panels horseshoe vortices and control points 
    # --------------------------------------------------------------------------------------- 
    y_a   = y_coordinates[:-1] 
    y_b   = y_coordinates[1:]             
    del_y = y_coordinates[1:] - y_coordinates[:-1] 
    
    # Let relevant control surfaces know which y-coords they are required to have----------------------------------
    if not wing.is_a_control_surface:
        i_break = 0
        for idx_y in range(n_sw):
            span_break = span_breaks[i_break]
            cs_IDs     = span_break.cs_IDs[:,1] #only the outboard control surfaces
            y_coord    = y_coordinates[idx_y]
            
            for cs_ID in cs_IDs[cs_IDs >= 0]:
                cs_tag     = wing.tag + '__cs_id_{}'.format(int(cs_ID))
                cs_wing    = wings[cs_tag]
                rel_offset = cs_wing.origin[0,1] - wing.origin[0][1] if not vertical_wing else cs_wing.origin[0,2] - wing.origin[0][2]
                cs_wing.y_coords_required.append(y_coord - rel_offset)
            
            if y_coordinates[idx_y+1] == break_spans[i_break+1]: 
                i_break += 1
    
    # -------------------------------------------------------------------------------------------------------------
    # Run the strip contruction loop again if wing is symmetric. 
    # Reflection plane = x-y plane for vertical wings. Otherwise, reflection plane = x-z plane
    signs         = rp.array([1, -1]) # acts as a multiplier for symmetry. -1 is only ever used for symmetric wings
    symmetry_mask = [True,sym_para]
    for sym_sign in signs[symmetry_mask]:
        # create empty vectors for coordinates 
        xah   = rp.zeros(n_cw*n_sw)
        yah   = rp.zeros(n_cw*n_sw)
        zah   = rp.zeros(n_cw*n_sw)
        xbh   = rp.zeros(n_cw*n_sw)
        ybh   = rp.zeros(n_cw*n_sw)
        zbh   = rp.zeros(n_cw*n_sw)    
        xch   = rp.zeros(n_cw*n_sw)
        ych   = rp.zeros(n_cw*n_sw)
        zch   = rp.zeros(n_cw*n_sw)    
        xa1   = rp.zeros(n_cw*n_sw)
        ya1   = rp.zeros(n_cw*n_sw)
        za1   = rp.zeros(n_cw*n_sw)
        xa2   = rp.zeros(n_cw*n_sw)
        ya2   = rp.zeros(n_cw*n_sw)
        za2   = rp.zeros(n_cw*n_sw)    
        xb1   = rp.zeros(n_cw*n_sw)
        yb1   = rp.zeros(n_cw*n_sw)
        zb1   = rp.zeros(n_cw*n_sw)
        xb2   = rp.zeros(n_cw*n_sw) 
        yb2   = rp.zeros(n_cw*n_sw) 
        zb2   = rp.zeros(n_cw*n_sw)    
        xac   = rp.zeros(n_cw*n_sw)
        yac   = rp.zeros(n_cw*n_sw)
        zac   = rp.zeros(n_cw*n_sw)    
        xbc   = rp.zeros(n_cw*n_sw)
        ybc   = rp.zeros(n_cw*n_sw)
        zbc   = rp.zeros(n_cw*n_sw)  
        xc    = rp.zeros(n_cw*n_sw) 
        yc    = rp.zeros(n_cw*n_sw) 
        zc    = rp.zeros(n_cw*n_sw) 
        x     = rp.zeros((n_cw+1)*(n_sw+1)) # may have to change to make space for split if control surfaces are allowed to have more than two Segments
        y     = rp.zeros((n_cw+1)*(n_sw+1)) 
        z     = rp.zeros((n_cw+1)*(n_sw+1))         
        cs_w  = rp.zeros(n_sw)        
        
        # adjust origin for symmetry with special case for vertical symmetry
        wing_origin_x = wing_origin[0]
        wing_origin_y = wing_origin[1] * ((1-vertical_wing)*sym_sign+vertical_wing)
        wing_origin_z = wing_origin[2] * ((1-vertical_wing)+sym_sign*vertical_wing)
            
        # ---------------------------------------------------------------------------------------------------------
        # Loop over each strip of panels in the wing
        i_break = 0           
        for idx_y in range(n_sw):
            # define basic geometric values------------------------------------------------------------------------
            # inboard, outboard, and central panel values
            eta_a = (y_a[idx_y] - break_spans[i_break])  
            eta_b = (y_b[idx_y] - break_spans[i_break]) 
            eta   = (y_b[idx_y] - del_y[idx_y]/2 - break_spans[i_break]) 
            
            # Inverted wing
            wing.inverted_wing = -rp.sign(break_dihedral[i_break] - rp.pi/2)
    
            segment_chord_ratio = (break_chord[i_break+1] - break_chord[i_break])/section_span[i_break+1]
            segment_twist_ratio = (break_twist[i_break+1] - break_twist[i_break])/section_span[i_break+1]
    
            wing_chord_section_a  = break_chord[i_break] + (eta_a*segment_chord_ratio) 
            wing_chord_section_b  = break_chord[i_break] + (eta_b*segment_chord_ratio)
            wing_chord_section    = break_chord[i_break] + (eta*segment_chord_ratio)
    
            # x-positions based on whether the wing needs 'cuts' for its control sufaces
            nondim_x_stations = rp.interp(rp.linspace(0.,1.,num=n_cw+1), rp.array([0.,1.]), rp.array([section_LE_cut[i_break], section_TE_cut[i_break]]))
            x_stations_a      = nondim_x_stations * wing_chord_section_a  #x positions accounting for control surface cuts, relative to leading
            x_stations_b      = nondim_x_stations * wing_chord_section_b
            x_stations        = nondim_x_stations * wing_chord_section
            
            delta_x_a = (x_stations_a[-1] - x_stations_a[0])/n_cw  
            delta_x_b = (x_stations_b[-1] - x_stations_b[0])/n_cw      
            delta_x   = (x_stations[-1]   - x_stations[0]  )/n_cw             
    
            # define coordinates of horseshoe vortices and control points------------------------------------------
            xi_a1 = break_x_offset[i_break] + eta_a*rp.tan(break_sweep[i_break]) + x_stations_a[:-1]                  # x coordinate of top left corner of panel
            xi_ah = break_x_offset[i_break] + eta_a*rp.tan(break_sweep[i_break]) + x_stations_a[:-1] + delta_x_a*0.25 # x coordinate of left corner of panel
            xi_ac = break_x_offset[i_break] + eta_a*rp.tan(break_sweep[i_break]) + x_stations_a[:-1] + delta_x_a*0.75 # x coordinate of bottom left corner of control point vortex  
            xi_a2 = break_x_offset[i_break] + eta_a*rp.tan(break_sweep[i_break]) + x_stations_a[1:]                   # x coordinate of bottom left corner of bound vortex 
            xi_b1 = break_x_offset[i_break] + eta_b*rp.tan(break_sweep[i_break]) + x_stations_b[:-1]                  # x coordinate of top right corner of panel      
            xi_bh = break_x_offset[i_break] + eta_b*rp.tan(break_sweep[i_break]) + x_stations_b[:-1] + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
            xi_bc = break_x_offset[i_break] + eta_b*rp.tan(break_sweep[i_break]) + x_stations_b[:-1] + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
            xi_b2 = break_x_offset[i_break] + eta_b*rp.tan(break_sweep[i_break]) + x_stations_b[1:]                   # x coordinate of bottom right corner of panel
            xi_ch = break_x_offset[i_break] + eta  *rp.tan(break_sweep[i_break]) + x_stations[:-1]   + delta_x  *0.25 # x coordinate center of bound vortex of each panel 
            xi_c  = break_x_offset[i_break] + eta  *rp.tan(break_sweep[i_break]) + x_stations[:-1]   + delta_x  *0.75 # x coordinate three-quarter chord control point for each panel
    
            #adjust for camber-------------------------------------------------------------------------------------    
            #format camber vars for wings vs control surface wings
            nondim_camber_x_coords = break_camber_xs[i_break] *1
            nondim_camber          = break_camber_zs[i_break] *1
            if wing.is_a_control_surface: #rescale so that airfoils get cut properly
                if not wing.is_slat:
                    nondim_camber_x_coords = nondim_camber_x_coords - 1 - wing.chord_fraction
                nondim_camber_x_coords = nondim_camber_x_coords/wing.chord_fraction
                nondim_camber          = nondim_camber/ wing.chord_fraction
    
            # adjustment of coordinates for camber
            section_camber_a  = nondim_camber*wing_chord_section_a  
            section_camber_b  = nondim_camber*wing_chord_section_b  
            section_camber_c  = nondim_camber*wing_chord_section             
            
            section_x_coord_a = nondim_camber_x_coords*wing_chord_section_a
            section_x_coord_b = nondim_camber_x_coords*wing_chord_section_b
            section_x_coord   = nondim_camber_x_coords*wing_chord_section
    
            z_c_a1 = rp.interp((x_stations_a[:-1]                 ) ,section_x_coord_a, section_camber_a) 
            z_c_ah = rp.interp((x_stations_a[:-1] + delta_x_a*0.25) ,section_x_coord_a, section_camber_a)
            z_c_ac = rp.interp((x_stations_a[:-1] + delta_x_a*0.75) ,section_x_coord_a, section_camber_a) 
            z_c_a2 = rp.interp((x_stations_a[1:]                  ) ,section_x_coord_a, section_camber_a) 
            z_c_b1 = rp.interp((x_stations_b[:-1]                 ) ,section_x_coord_b, section_camber_b)   
            z_c_bh = rp.interp((x_stations_b[:-1] + delta_x_b*0.25) ,section_x_coord_b, section_camber_b) 
            z_c_bc = rp.interp((x_stations_b[:-1] + delta_x_b*0.75) ,section_x_coord_b, section_camber_b) 
            z_c_b2 = rp.interp((x_stations_b[1:]                  ) ,section_x_coord_b, section_camber_b) 
            z_c_ch = rp.interp((x_stations[:-1]   + delta_x  *0.25) ,section_x_coord  , section_camber_c) 
            z_c    = rp.interp((x_stations[:-1]   + delta_x  *0.75) ,section_x_coord  , section_camber_c) 
    
            # adjust for dihedral and add to camber----------------------------------------------------------------    
            zeta_a1 = break_z_offset[i_break] + eta_a*rp.tan(break_dihedral[i_break])  + z_c_a1  # z coordinate of top left corner of panel
            zeta_ah = break_z_offset[i_break] + eta_a*rp.tan(break_dihedral[i_break])  + z_c_ah  # z coordinate of left corner of bound vortex  
            zeta_a2 = break_z_offset[i_break] + eta_a*rp.tan(break_dihedral[i_break])  + z_c_a2  # z coordinate of bottom left corner of panel
            zeta_ac = break_z_offset[i_break] + eta_a*rp.tan(break_dihedral[i_break])  + z_c_ac  # z coordinate of bottom left corner of panel of control point
            zeta_bc = break_z_offset[i_break] + eta_b*rp.tan(break_dihedral[i_break])  + z_c_bc  # z coordinate of top right corner of panel of control point                          
            zeta_b1 = break_z_offset[i_break] + eta_b*rp.tan(break_dihedral[i_break])  + z_c_b1  # z coordinate of top right corner of panel  
            zeta_bh = break_z_offset[i_break] + eta_b*rp.tan(break_dihedral[i_break])  + z_c_bh  # z coordinate of right corner of bound vortex        
            zeta_b2 = break_z_offset[i_break] + eta_b*rp.tan(break_dihedral[i_break])  + z_c_b2  # z coordinate of bottom right corner of panel                 
            zeta_ch = break_z_offset[i_break] + eta  *rp.tan(break_dihedral[i_break])  + z_c_ch  # z coordinate center of bound vortex on each panel
            zeta    = break_z_offset[i_break] + eta  *rp.tan(break_dihedral[i_break])  + z_c     # z coordinate three-quarter chord control point for each panel
    
            # adjust for twist-------------------------------------------------------------------------------------
            # pivot point is the leading edge before camber  
            pivot_x_a = break_x_offset[i_break] + eta_a*rp.tan(break_sweep[i_break])             # x location of leading edge left corner of wing
            pivot_x_b = break_x_offset[i_break] + eta_b*rp.tan(break_sweep[i_break])             # x location of leading edge right of wing
            pivot_x   = break_x_offset[i_break] + eta  *rp.tan(break_sweep[i_break])             # x location of leading edge center of wing
            
            pivot_z_a = break_z_offset[i_break] + eta_a*rp.tan(break_dihedral[i_break])          # z location of leading edge left corner of wing
            pivot_z_b = break_z_offset[i_break] + eta_b*rp.tan(break_dihedral[i_break])          # z location of leading edge right of wing
            pivot_z   = break_z_offset[i_break] + eta  *rp.tan(break_dihedral[i_break])          # z location of leading edge center of wing
    
            # adjust twist pivot line for control surface wings: offset leading edge to match that of the owning wing            
            if wing.is_a_control_surface and not wing.is_slat: #correction only leading for non-leading edge control surfaces since the LE is the pivot by default
                nondim_cs_LE = (1 - wing.chord_fraction)
                pivot_x_a   = pivot_x_a - nondim_cs_LE *(wing_chord_section_a /wing.chord_fraction) 
                pivot_x_b   = pivot_x_b - nondim_cs_LE *(wing_chord_section_b /wing.chord_fraction) 
                pivot_x     = pivot_x - nondim_cs_LE *(wing_chord_section   /wing.chord_fraction) 
    
            # adjust coordinates for twist
            section_twist_a = break_twist[i_break] + (eta_a * segment_twist_ratio)               # twist at left side of panel
            section_twist_b = break_twist[i_break] + (eta_b * segment_twist_ratio)               # twist at right side of panel
            section_twist   = break_twist[i_break] + (eta   * segment_twist_ratio)               # twist at center local chord 
    
            xi_prime_a1    = pivot_x_a + rp.cos(section_twist_a)*(xi_a1-pivot_x_a) + rp.sin(section_twist_a)*(zeta_a1-pivot_z_a) # x coordinate transformation of top left corner
            xi_prime_ah    = pivot_x_a + rp.cos(section_twist_a)*(xi_ah-pivot_x_a) + rp.sin(section_twist_a)*(zeta_ah-pivot_z_a) # x coordinate transformation of bottom left corner
            xi_prime_ac    = pivot_x_a + rp.cos(section_twist_a)*(xi_ac-pivot_x_a) + rp.sin(section_twist_a)*(zeta_a2-pivot_z_a) # x coordinate transformation of bottom left corner of control point
            xi_prime_a2    = pivot_x_a + rp.cos(section_twist_a)*(xi_a2-pivot_x_a) + rp.sin(section_twist_a)*(zeta_a2-pivot_z_a) # x coordinate transformation of bottom left corner
            xi_prime_b1    = pivot_x_b + rp.cos(section_twist_b)*(xi_b1-pivot_x_b) + rp.sin(section_twist_b)*(zeta_b1-pivot_z_b) # x coordinate transformation of top right corner 
            xi_prime_bh    = pivot_x_b + rp.cos(section_twist_b)*(xi_bh-pivot_x_b) + rp.sin(section_twist_b)*(zeta_bh-pivot_z_b) # x coordinate transformation of top right corner 
            xi_prime_bc    = pivot_x_b + rp.cos(section_twist_b)*(xi_bc-pivot_x_b) + rp.sin(section_twist_b)*(zeta_b1-pivot_z_b) # x coordinate transformation of top right corner of control point                         
            xi_prime_b2    = pivot_x_b + rp.cos(section_twist_b)*(xi_b2-pivot_x_b) + rp.sin(section_twist_b)*(zeta_b2-pivot_z_b) # x coordinate transformation of botton right corner 
            xi_prime_ch    = pivot_x   + rp.cos(section_twist)  *(xi_ch-pivot_x)   + rp.sin(section_twist)  *(zeta_ch-pivot_z)   # x coordinate transformation of center of horeshoe vortex 
            xi_prime       = pivot_x   + rp.cos(section_twist)  *(xi_c -pivot_x)   + rp.sin(section_twist)  *(zeta   -pivot_z)   # x coordinate transformation of control point
    
            zeta_prime_a1  = pivot_z_a - rp.sin(section_twist_a)*(xi_a1-pivot_x_a) + rp.cos(section_twist_a)*(zeta_a1-pivot_z_a) # z coordinate transformation of top left corner
            zeta_prime_ah  = pivot_z_a - rp.sin(section_twist_a)*(xi_ah-pivot_x_a) + rp.cos(section_twist_a)*(zeta_ah-pivot_z_a) # z coordinate transformation of bottom left corner
            zeta_prime_ac  = pivot_z_a - rp.sin(section_twist_a)*(xi_ac-pivot_x_a) + rp.cos(section_twist_a)*(zeta_ac-pivot_z_a) # z coordinate transformation of bottom left corner
            zeta_prime_a2  = pivot_z_a - rp.sin(section_twist_a)*(xi_a2-pivot_x_a) + rp.cos(section_twist_a)*(zeta_a2-pivot_z_a) # z coordinate transformation of bottom left corner
            zeta_prime_b1  = pivot_z_b - rp.sin(section_twist_b)*(xi_b1-pivot_x_b) + rp.cos(section_twist_b)*(zeta_b1-pivot_z_b) # z coordinate transformation of top right corner 
            zeta_prime_bh  = pivot_z_b - rp.sin(section_twist_b)*(xi_bh-pivot_x_b) + rp.cos(section_twist_b)*(zeta_bh-pivot_z_b) # z coordinate transformation of top right corner 
            zeta_prime_bc  = pivot_z_b - rp.sin(section_twist_b)*(xi_bc-pivot_x_b) + rp.cos(section_twist_b)*(zeta_bc-pivot_z_b) # z coordinate transformation of top right corner                         
            zeta_prime_b2  = pivot_z_b - rp.sin(section_twist_b)*(xi_b2-pivot_x_b) + rp.cos(section_twist_b)*(zeta_b2-pivot_z_b) # z coordinate transformation of botton right corner 
            zeta_prime_ch  = pivot_z   - rp.sin(section_twist)  *(xi_ch-pivot_x)   + rp.cos(-section_twist) *(zeta_ch-pivot_z)   # z coordinate transformation of center of horseshoe
            zeta_prime     = pivot_z   - rp.sin(section_twist)  *(xi_c -pivot_x)   + rp.cos(-section_twist) *(zeta   -pivot_z)   # z coordinate transformation of control point
            
            # Define y-coordinate and other arrays-----------------------------------------------------------------
            # take normal value for first wing, then reflect over xz plane for a symmetric wing
            y_prime_as = (rp.ones(n_cw+1)*y_a[idx_y]                 ) *sym_sign          
            y_prime_a1 = (y_prime_as[:-1]                            ) *1       
            y_prime_ah = (y_prime_as[:-1]                            ) *1       
            y_prime_ac = (y_prime_as[:-1]                            ) *1          
            y_prime_a2 = (y_prime_as[:-1]                            ) *1        
            y_prime_bs = (rp.ones(n_cw+1)*y_b[idx_y]                 ) *sym_sign            
            y_prime_b1 = (y_prime_bs[:-1]                            ) *1         
            y_prime_bh = (y_prime_bs[:-1]                            ) *1         
            y_prime_bc = (y_prime_bs[:-1]                            ) *1         
            y_prime_b2 = (y_prime_bs[:-1]                            ) *1   
            y_prime_ch = (rp.ones(n_cw)*(y_b[idx_y] - del_y[idx_y]/2)) *sym_sign
            y_prime    = (y_prime_ch                                 ) *1    
            
            # populate all corners of all panels. Right side only populated for last strip wing the wing
            xi_prime_as   = rp.concatenate([xi_prime_a1,  rp.array([xi_prime_a2  [-1]])])*1
            xi_prime_bs   = rp.concatenate([xi_prime_b1,  rp.array([xi_prime_b2  [-1]])])*1
            zeta_prime_as = rp.concatenate([zeta_prime_a1,rp.array([zeta_prime_a2[-1]])])*1            
            zeta_prime_bs = rp.concatenate([zeta_prime_b1,rp.array([zeta_prime_b2[-1]])])*1  
            
            # reflect over the plane y = z for a vertical wing-----------------------------------------------------
            if vertical_wing:
                y_prime_a1, zeta_prime_a1 = zeta_prime_a1, wing.inverted_wing*y_prime_a1
                y_prime_ah, zeta_prime_ah = zeta_prime_ah, wing.inverted_wing*y_prime_ah
                y_prime_ac, zeta_prime_ac = zeta_prime_ac, wing.inverted_wing*y_prime_ac
                y_prime_a2, zeta_prime_a2 = zeta_prime_a2, wing.inverted_wing*y_prime_a2
                                                                     
                y_prime_b1, zeta_prime_b1 = zeta_prime_b1, wing.inverted_wing*y_prime_b1
                y_prime_bh, zeta_prime_bh = zeta_prime_bh, wing.inverted_wing*y_prime_bh
                y_prime_bc, zeta_prime_bc = zeta_prime_bc, wing.inverted_wing*y_prime_bc
                y_prime_b2, zeta_prime_b2 = zeta_prime_b2, wing.inverted_wing*y_prime_b2
                
                y_prime_ch, zeta_prime_ch = zeta_prime_ch, wing.inverted_wing*y_prime_ch
                y_prime   , zeta_prime    = zeta_prime   , wing.inverted_wing*y_prime
                
                y_prime_as, zeta_prime_as = zeta_prime_as, wing.inverted_wing*y_prime_as

 
                y_prime_bs = wing.inverted_wing*y_prime_bs
                y_prime_bs, zeta_prime_bs = zeta_prime_bs, y_prime_bs
                 
            # store coordinates of panels, horseshoeces vortices and control points relative to wing root----------
            xa1 = xa1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_a1) # top left corner of panel
            ya1 = ya1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_a1)
            za1 = za1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_a1)
            xah = xah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_ah) # left coord of horseshoe
            yah = yah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_ah)
            zah = zah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_ah)
            xac = xac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_ac) # left coord of control point
            yac = yac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_ac)
            zac = zac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_ac)
            xa2 = xa2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_a2) # bottom left corner of panel
            ya2 = ya2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_a2)
            za2 = za2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_a2)
                                             
            xb1 = xb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_b1) # top right corner of panel
            yb1 = yb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_b1)
            zb1 = zb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_b1)
            xbh = xbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_bh) # right coord of horseshoe
            ybh = ybh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_bh)
            zbh = zbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_bh)
            xbc = xbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_bc) # right coord of control point
            ybc = ybc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_bc)
            zbc = zbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_bc)
            xb2 = xb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_b2) # bottom right corner of panel
            yb2 = yb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_b2)
            zb2 = zb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_b2)
                                             
            xch = xch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime_ch) # center coord of horseshoe
            ych = ych.at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime_ch)
            zch = zch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime_ch)
            xc  = xc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(xi_prime) # center (true) coord of control point
            yc  = yc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(y_prime)
            zc  = zc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(zeta_prime)
           
            x = x.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(xi_prime_as) # x, y, z represent all all points of the corners of the panels, LE and TE inclusive
            y = y.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(y_prime_as) # the final right corners get appended at last strip in wing, later
            z = z.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(zeta_prime_as)

            cs_w = cs_w.at[idx_y].set(wing_chord_section)     
                   
            # store this strip's discretization information--------------------------------------------------------
            LE_inds        = rp.full((n_cw,), False)
            TE_inds        = rp.full((n_cw,), False)
            LE_inds = LE_inds.at[0].set(True)
            TE_inds = TE_inds.at[-1].set(True)
            
            RNMAX          = rp.ones(n_cw, rp.int16)*n_cw
            panel_numbers  = rp.linspace(1,n_cw,n_cw, dtype=rp.int16)           
                        
            is_a_slat                  = wing.is_a_control_surface and wing.is_slat
            strip_has_no_slat          = (not wing.is_a_control_surface) and (span_breaks[i_break].cs_IDs[0,1] == -1) # wing's le, outboard control surface ID
            exposed_leading_edge_flag  = 1 if is_a_slat or strip_has_no_slat else 0  
            
            VD.leading_edge_indices      = rp.append(VD.leading_edge_indices     , LE_inds                  ) 
            VD.trailing_edge_indices     = rp.append(VD.trailing_edge_indices    , TE_inds                  )            
            VD.panels_per_strip          = rp.append(VD.panels_per_strip         , RNMAX                    )
            VD.chordwise_panel_number    = rp.append(VD.chordwise_panel_number   , panel_numbers            )  
            VD.exposed_leading_edge_flag = rp.append(VD.exposed_leading_edge_flag, exposed_leading_edge_flag)
            
            #increment i_break if needed; check for end of wing----------------------------------------------------
            if y_b[idx_y] == break_spans[i_break+1]: 
                i_break += 1            
        #End 'for each strip' loop    
        
        # store outboardmost edge
        x = x.at[-(n_cw+1):].set(xi_prime_bs)
        y = y.at[-(n_cw+1):].set(y_prime_bs)
        z = z.at[-(n_cw+1):].set(zeta_prime_bs)
        
        # adjusting coordinate axis so reference point is at the nose of the aircraft------------------------------
        xah = xah + wing_origin_x # x coordinate of left corner of bound vortex 
        yah = yah + wing_origin_y # y coordinate of left corner of bound vortex 
        zah = zah + wing_origin_z # z coordinate of left corner of bound vortex 
        xbh = xbh + wing_origin_x # x coordinate of right corner of bound vortex 
        ybh = ybh + wing_origin_y # y coordinate of right corner of bound vortex 
        zbh = zbh + wing_origin_z # z coordinate of right corner of bound vortex 
        xch = xch + wing_origin_x # x coordinate of center of bound vortex on panel
        ych = ych + wing_origin_y # y coordinate of center of bound vortex on panel
        zch = zch + wing_origin_z # z coordinate of center of bound vortex on panel  
    
        xa1 = xa1 + wing_origin_x # x coordinate of top left corner of panel
        ya1 = ya1 + wing_origin_y # y coordinate of bottom left corner of panel
        za1 = za1 + wing_origin_z # z coordinate of top left corner of panel
        xa2 = xa2 + wing_origin_x # x coordinate of bottom left corner of panel
        ya2 = ya2 + wing_origin_y # x coordinate of bottom left corner of panel
        za2 = za2 + wing_origin_z # z coordinate of bottom left corner of panel  
    
        xb1 = xb1 + wing_origin_x # x coordinate of top right corner of panel  
        yb1 = yb1 + wing_origin_y # y coordinate of top right corner of panel 
        zb1 = zb1 + wing_origin_z # z coordinate of top right corner of panel 
        xb2 = xb2 + wing_origin_x # x coordinate of bottom rightcorner of panel 
        yb2 = yb2 + wing_origin_y # y coordinate of bottom rightcorner of panel 
        zb2 = zb2 + wing_origin_z # z coordinate of bottom right corner of panel                   
    
        xac = xac + wing_origin_x  # x coordinate of control points on panel
        yac = yac + wing_origin_y  # y coordinate of control points on panel
        zac = zac + wing_origin_z  # z coordinate of control points on panel
        xbc = xbc + wing_origin_x  # x coordinate of control points on panel
        ybc = ybc + wing_origin_y  # y coordinate of control points on panel
        zbc = zbc + wing_origin_z  # z coordinate of control points on panel
    
        xc  = xc  + wing_origin_x  # x coordinate of control points on panel
        yc  = yc  + wing_origin_y  # y coordinate of control points on panel
        zc  = zc  + wing_origin_z  # y coordinate of control points on panel
        x   = x   + wing_origin_x  # x coordinate of control points on panel
        y   = y   + wing_origin_y  # y coordinate of control points on panel
        z   = z   + wing_origin_z  # y coordinate of control points on panel
        
        # VD discretization information----------------------------------------------------------------------------
        
        # increment number of wings and panels
        n_panels = len(xch)
        VD.n_w  = VD.n_w + 1             
        VD.n_cp = VD.n_cp + n_panels 
        
        # store this wing's discretization information  
        first_panel_ind  = VD.XAH.size
        first_strip_ind  = VD.chordwise_breaks.size
        chordwise_breaks = first_panel_ind + rp.arange(n_panels)[0::n_cw]
        ID               = VD.counter*1
        
        VD.chordwise_breaks = rp.append(VD.chordwise_breaks, chordwise_breaks)
        VD.spanwise_breaks  = rp.append(VD.spanwise_breaks , first_strip_ind )         
        VD.n_sw             = rp.append(VD.n_sw            , n_sw            )
        VD.n_cw             = rp.append(VD.n_cw            , n_cw            )
        VD.surface_ID       = rp.append(VD.surface_ID      , rp.ones(n_cw*n_sw)*ID*sym_sign) # Update me when the loop is gone
        VD.surface_ID_full  = rp.append(VD.surface_ID_full , rp.ones((n_cw+1)*(n_sw+1))*ID*sym_sign) # Update me when the loop is gone    
                
        # ---------------------------------------------------------------------------------------
        # STEP 7: Store wing in vehicle vector
        # --------------------------------------------------------------------------------------- 
        VD.XAH    = rp.append(VD.XAH  , rp.array(xah  , dtype=precision))
        VD.YAH    = rp.append(VD.YAH  , rp.array(yah  , dtype=precision))
        VD.ZAH    = rp.append(VD.ZAH  , rp.array(zah  , dtype=precision))
        VD.XBH    = rp.append(VD.XBH  , rp.array(xbh  , dtype=precision))
        VD.YBH    = rp.append(VD.YBH  , rp.array(ybh  , dtype=precision))
        VD.ZBH    = rp.append(VD.ZBH  , rp.array(zbh  , dtype=precision))
        VD.XCH    = rp.append(VD.XCH  , rp.array(xch  , dtype=precision))
        VD.YCH    = rp.append(VD.YCH  , rp.array(ych  , dtype=precision))
        VD.ZCH    = rp.append(VD.ZCH  , rp.array(zch  , dtype=precision))            
        VD.XA1    = rp.append(VD.XA1  , rp.array(xa1  , dtype=precision))
        VD.YA1    = rp.append(VD.YA1  , rp.array(ya1  , dtype=precision))
        VD.ZA1    = rp.append(VD.ZA1  , rp.array(za1  , dtype=precision))
        VD.XA2    = rp.append(VD.XA2  , rp.array(xa2  , dtype=precision))
        VD.YA2    = rp.append(VD.YA2  , rp.array(ya2  , dtype=precision))
        VD.ZA2    = rp.append(VD.ZA2  , rp.array(za2  , dtype=precision))        
        VD.XB1    = rp.append(VD.XB1  , rp.array(xb1  , dtype=precision))
        VD.YB1    = rp.append(VD.YB1  , rp.array(yb1  , dtype=precision))
        VD.ZB1    = rp.append(VD.ZB1  , rp.array(zb1  , dtype=precision))
        VD.XB2    = rp.append(VD.XB2  , rp.array(xb2  , dtype=precision))                
        VD.YB2    = rp.append(VD.YB2  , rp.array(yb2  , dtype=precision))        
        VD.ZB2    = rp.append(VD.ZB2  , rp.array(zb2  , dtype=precision)) 
        VD.XAC    = rp.append(VD.XAC  , rp.array(xac  , dtype=precision))
        VD.YAC    = rp.append(VD.YAC  , rp.array(yac  , dtype=precision)) 
        VD.ZAC    = rp.append(VD.ZAC  , rp.array(zac  , dtype=precision)) 
        VD.XBC    = rp.append(VD.XBC  , rp.array(xbc  , dtype=precision))
        VD.YBC    = rp.append(VD.YBC  , rp.array(ybc  , dtype=precision)) 
        VD.ZBC    = rp.append(VD.ZBC  , rp.array(zbc  , dtype=precision))  
        VD.XC     = rp.append(VD.XC   , rp.array(xc   , dtype=precision))
        VD.YC     = rp.append(VD.YC   , rp.array(yc   , dtype=precision))
        VD.ZC     = rp.append(VD.ZC   , rp.array(zc   , dtype=precision))  
        VD.X      = rp.append(VD.X    , rp.array(x    , dtype=precision))
        VD.Y      = rp.append(VD.Y    , rp.array(y    , dtype=precision))
        VD.Z      = rp.append(VD.Z    , rp.array(z    , dtype=precision))         
        VD.CS     = rp.append(VD.CS   , rp.array(cs_w , dtype=precision)) 
        VD.DY     = rp.append(VD.DY   , rp.array(del_y, dtype=precision))    
    #End symmetry loop
    
    VD.symmetric_wings = rp.append(VD.symmetric_wings, int(sym_para))
    
    # Pack wing data
    wing.n_sw = n_sw
    wing.n_cw = n_cw    
    
    return VD, wing
    

# ----------------------------------------------------------------------
#  Discretize Fuselage
# ----------------------------------------------------------------------
def generate_fuselage_and_nacelle_vortex_distribution(VD,fus,n_cw,n_sw,precision,model_geometry=False):
    """ This generates the vortex distribution points on a fuselage or nacelle component
    Assumptions: 
    If nacelle has segments defined, the mean width and height of the nacelle is used
    Source:   
    None
    
    Inputs:   
    VD                   - vortex distribution
    
    Properties Used:
    N/A
    """    
    
    fhs_xa1 = rp.zeros(n_cw*n_sw)
    fhs_ya1 = rp.zeros(n_cw*n_sw)
    fhs_za1 = rp.zeros(n_cw*n_sw)
    fhs_xa2 = rp.zeros(n_cw*n_sw)
    fhs_ya2 = rp.zeros(n_cw*n_sw)
    fhs_za2 = rp.zeros(n_cw*n_sw)
    fhs_xb1 = rp.zeros(n_cw*n_sw)
    fhs_yb1 = rp.zeros(n_cw*n_sw)
    fhs_zb1 = rp.zeros(n_cw*n_sw)
    fhs_yb2 = rp.zeros(n_cw*n_sw)
    fhs_xb2 = rp.zeros(n_cw*n_sw)
    fhs_zb2 = rp.zeros(n_cw*n_sw)
    fhs_xah = rp.zeros(n_cw*n_sw)
    fhs_yah = rp.zeros(n_cw*n_sw)
    fhs_zah = rp.zeros(n_cw*n_sw)
    fhs_xbh = rp.zeros(n_cw*n_sw)
    fhs_ybh = rp.zeros(n_cw*n_sw)
    fhs_zbh = rp.zeros(n_cw*n_sw)
    fhs_xch = rp.zeros(n_cw*n_sw)
    fhs_ych = rp.zeros(n_cw*n_sw)
    fhs_zch = rp.zeros(n_cw*n_sw)
    fhs_xc  = rp.zeros(n_cw*n_sw)
    fhs_yc  = rp.zeros(n_cw*n_sw)
    fhs_zc  = rp.zeros(n_cw*n_sw)
    fhs_xac = rp.zeros(n_cw*n_sw)
    fhs_yac = rp.zeros(n_cw*n_sw)
    fhs_zac = rp.zeros(n_cw*n_sw)
    fhs_xbc = rp.zeros(n_cw*n_sw)
    fhs_ybc = rp.zeros(n_cw*n_sw)
    fhs_zbc = rp.zeros(n_cw*n_sw)
    fhs_x   = rp.zeros((n_cw+1)*(n_sw+1))
    fhs_y   = rp.zeros((n_cw+1)*(n_sw+1))
    fhs_z   = rp.zeros((n_cw+1)*(n_sw+1))      

    fvs_xc    = rp.zeros(n_cw*n_sw)
    fvs_zc    = rp.zeros(n_cw*n_sw)
    fvs_yc    = rp.zeros(n_cw*n_sw)   
    fvs_x     = rp.zeros((n_cw+1)*(n_sw+1))
    fvs_y     = rp.zeros((n_cw+1)*(n_sw+1))
    fvs_z     = rp.zeros((n_cw+1)*(n_sw+1))   
    fus_v_cs  = rp.zeros(n_sw)     
    
    # arrays to hold strip discretization values
    leading_edge_indices    = rp.array([],dtype=bool)    
    trailing_edge_indices   = rp.array([],dtype=bool)    
    panels_per_strip        = rp.array([],dtype=rp.int16)
    chordwise_panel_number  = rp.array([],dtype=rp.int16)               

    # geometry values
    origin     = fus.origin[0]

    # --TO DO-- model fuselage segments if defined, else use the following code
    
    # Horizontal Sections of fuselage
    fhs        = Data()        
    fhs.origin = rp.zeros((n_sw+1,3))        
    fhs.chord  = rp.zeros((n_sw+1))         
    fhs.sweep  = rp.zeros((n_sw+1))     
                 
    fvs        = Data() 
    fvs.origin = rp.zeros((n_sw+1,3))
    fvs.chord  = rp.zeros((n_sw+1)) 
    fvs.sweep  = rp.zeros((n_sw+1))

    if isinstance(fus, RCAIDE.Library.Components.Fuselages.Fuselage):

        # Compute the curvature of the nose/tail given fineness ratio. Curvature is derived from general quadratic equation
        # This method relates the fineness ratio to the quadratic curve formula via a spline fit interpolation
        vec1               = [2 , 1.5, 1.2 , 1]
        vec2               = [1  ,1.57 , 3.2,  8]
        x                  = rp.linspace(0,1,4)
        fus_nose_curvature =  rp.interp(rp.interp(fus.fineness.nose,vec2,x), x , vec1)
        fus_tail_curvature =  rp.interp(rp.interp(fus.fineness.tail,vec2,x), x , vec1)
        semispan_h = fus.width * 0.5
        semispan_v = fus.heights.maximum * 0.5
        si         = rp.arange(1,((n_sw*2)+2))
        spacing    = rp.cos((2*si - 1)/(2*len(si))*rp.pi)
        h_array    = semispan_h*spacing[0:int((len(si)+1)/2)][::-1]
        v_array    = semispan_v*spacing[0:int((len(si)+1)/2)][::-1]

        for i in range(n_sw+1):
            fhs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
            fhs.nose_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
            fhs.tail_length   = ((1 - ((abs(h_array[i]/semispan_h))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
            fhs.nose_origin   = fus.lengths.nose - fhs.nose_length
            fhs.origin        = fhs.origin.at[i,:].set(rp.array([fhs.nose_origin , h_array[i], 0.])) # Local origin
            fhs.chord         = fhs.chord.at[i].set(fhs_cabin_length + fhs.nose_length + fhs.tail_length)

            fvs_cabin_length  = fus.lengths.total - (fus.lengths.nose + fus.lengths.tail)
            fvs.nose_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_nose_curvature ))**(1/fus_nose_curvature))*fus.lengths.nose
            fvs.tail_length   = ((1 - ((abs(v_array[i]/semispan_v))**fus_tail_curvature ))**(1/fus_tail_curvature))*fus.lengths.tail
            fvs.nose_origin   = fus.lengths.nose - fvs.nose_length
            fvs.origin        = fvs.origin.at[i,:].set(rp.array([origin[0] + fvs.nose_origin , origin[1] , origin[2]+  v_array[i]]))
            fvs.chord         = fvs.chord.at[i].set(fvs_cabin_length + fvs.nose_length + fvs.tail_length)

        fhs.sweep = fhs.sweep.at[:].set(rp.concatenate([rp.arctan((fhs.origin[:,0][1:] - fhs.origin[:,0][:-1])/(fhs.origin[:,1][1:]  - fhs.origin[:,1][:-1])) ,rp.zeros(1)]))
        fvs.sweep = fvs.sweep.at[:].set(rp.concatenate([rp.arctan((fvs.origin[:,0][1:] - fvs.origin[:,0][:-1])/(fvs.origin[:,2][1:]  - fvs.origin[:,2][:-1])) ,rp.zeros(1)]))

    # ---------------------------------------------------------------------------------------
    # STEP 9: Define coordinates of panels horseshoe vortices and control points  
    # ---------------------------------------------------------------------------------------        
    fhs_eta_a = h_array[:-1] 
    fhs_eta_b = h_array[1:]            
    fhs_del_y = h_array[1:] - h_array[:-1]
    fhs_eta   = h_array[1:] - fhs_del_y/2

    fvs_eta_a = v_array[:-1] 
    fvs_eta_b = v_array[1:]                  
    fvs_del_y = v_array[1:] - v_array[:-1]
    fvs_eta   = v_array[1:] - fvs_del_y/2 

    fhs_cs = rp.concatenate([fhs.chord,fhs.chord])
    fvs_cs = rp.concatenate([fvs.chord,fvs.chord])
    
    fus_h_area = 0
    fus_v_area = 0    

    # define coordinates of horseshoe vortices and control points       
    for idx_y in range(n_sw):  
        idx_x = rp.arange(n_cw)

        # fuselage horizontal section 
        delta_x_a = fhs.chord[idx_y]/n_cw      
        delta_x_b = fhs.chord[idx_y + 1]/n_cw    
        delta_x   = (fhs.chord[idx_y]+fhs.chord[idx_y + 1])/(2*n_cw)

        fhs_xi_a1 = fhs.origin[idx_y][0] + delta_x_a*idx_x                    # x coordinate of top left corner of panel
        fhs_xi_ah = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # x coordinate of left corner of panel
        fhs_xi_a2 = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # x coordinate of bottom left corner of bound vortex 
        fhs_xi_ac = fhs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # x coordinate of bottom left corner of control point vortex  
        fhs_xi_b1 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x                  # x coordinate of top right corner of panel      
        fhs_xi_bh = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25 # x coordinate of right corner of bound vortex         
        fhs_xi_b2 = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b      # x coordinate of bottom right corner of panel
        fhs_xi_bc = fhs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75 # x coordinate of bottom right corner of control point vortex         
        fhs_xi_c  = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.75   # x coordinate three-quarter chord control point for each panel
        fhs_xi_ch = (fhs.origin[idx_y][0] + fhs.origin[idx_y+1][0])/2  + delta_x*idx_x + delta_x*0.25   # x coordinate center of bound vortex of each panel 


        fhs_xa1 = fhs_xa1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_a1                       + fus.origin[0][0])
        fhs_ya1 = fhs_ya1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_za1 = fhs_za1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xa2 = fhs_xa2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_a2                       + fus.origin[0][0])
        fhs_ya2 = fhs_ya2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_za2 = fhs_za2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xb1 = fhs_xb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_b1                       + fus.origin[0][0])
        fhs_yb1 = fhs_yb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])
        fhs_zb1 = fhs_zb1.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xb2 = fhs_xb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_b2                       + fus.origin[0][0])
        fhs_yb2 = fhs_yb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])
        fhs_zb2 = fhs_zb2.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xah = fhs_xah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ah                       + fus.origin[0][0])
        fhs_yah = fhs_yah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_zah = fhs_zah.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xbh = fhs_xbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_bh                       + fus.origin[0][0])
        fhs_ybh = fhs_ybh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])
        fhs_zbh = fhs_zbh.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xch = fhs_xch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ch                       + fus.origin[0][0])
        fhs_ych = fhs_ych.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1])
        fhs_zch = fhs_zch.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xc  = fhs_xc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_c                        + fus.origin[0][0])
        fhs_yc  = fhs_yc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta[idx_y]    + fus.origin[0][1])
        fhs_zc  = fhs_zc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xac = fhs_xac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_ac                       + fus.origin[0][0])
        fhs_yac = fhs_yac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_zac = fhs_zac.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_xbc = fhs_xbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(fhs_xi_bc                       + fus.origin[0][0])
        fhs_ybc = fhs_ybc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fhs_eta_b[idx_y]  + fus.origin[0][1])
        fhs_zbc = fhs_zbc.at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                  + fus.origin[0][2])
        fhs_x = fhs_x.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.concatenate([fhs_xi_a1,rp.array([fhs_xi_a2[-1]])]) + fus.origin[0][0])
        fhs_y = fhs_y.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.ones(n_cw+1)*fhs_eta_a[idx_y]  + fus.origin[0][1])
        fhs_z = fhs_z.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.zeros(n_cw+1)                  + fus.origin[0][2])

        # fuselage vertical section                      
        delta_x_a = fvs.chord[idx_y]/n_cw      
        delta_x_b = fvs.chord[idx_y + 1]/n_cw    
        delta_x   = (fvs.chord[idx_y]+fvs.chord[idx_y + 1])/(2*n_cw)   

        fvs_xi_a1 = fvs.origin[idx_y][0] + delta_x_a*idx_x                    # z coordinate of top left corner of panel
        fvs_xi_ah = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.25   # z coordinate of left corner of panel
        fvs_xi_a2 = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a        # z coordinate of bottom left corner of bound vortex 
        fvs_xi_ac = fvs.origin[idx_y][0] + delta_x_a*idx_x + delta_x_a*0.75   # z coordinate of bottom left corner of control point vortex  
        fvs_xi_b1 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x                    # z coordinate of top right corner of panel      
        fvs_xi_bh = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.25   # z coordinate of right corner of bound vortex         
        fvs_xi_b2 = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b        # z coordinate of bottom right corner of panel
        fvs_xi_bc = fvs.origin[idx_y+1][0] + delta_x_b*idx_x + delta_x_b*0.75   # z coordinate of bottom right corner of control point vortex         
        fvs_xi_c  = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.75     # z coordinate three-quarter chord control point for each panel
        fvs_xi_ch = (fvs.origin[idx_y][0] + fvs.origin[idx_y+1][0])/2 + delta_x *idx_x + delta_x*0.25     # z coordinate center of bound vortex of each panel 

        fvs_xc  = fvs_xc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(fvs_xi_c                       + fus.origin[0][0])
        fvs_zc  = fvs_zc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.ones(n_cw)*fvs_eta[idx_y]   + fus.origin[0][2])
        fvs_yc  = fvs_yc .at[idx_y*n_cw:(idx_y+1)*n_cw].set(rp.zeros(n_cw)                 + fus.origin[0][1])
        fvs_x = fvs_x.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.concatenate([fvs_xi_a1,rp.array([fvs_xi_a2[-1]])]) + fus.origin[0][0])
        fvs_z = fvs_z.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2])
        fvs_y = fvs_y.at[idx_y*(n_cw+1):(idx_y+1)*(n_cw+1)].set(rp.zeros(n_cw+1)                 + fus.origin[0][1])
        
        fus_h_area = fus_h_area + ((fhs.chord[idx_y]+fhs.chord[idx_y + 1])/2)*(fhs_eta_b[idx_y] - fhs_eta_a[idx_y])
        fus_v_area = fus_v_area + ((fvs.chord[idx_y]+fvs.chord[idx_y + 1])/2)*(fvs_eta_b[idx_y] - fvs_eta_a[idx_y])
        
        # store this strip's discretization information
        LE_inds        = rp.full((n_cw,), 0)
        TE_inds        = rp.full((n_cw,), 0)
        LE_inds = LE_inds.at[0].set(True) 
        TE_inds = TE_inds.at[-1].set(True)
        
        RNMAX          = rp.ones(n_cw, rp.int16)*n_cw
        panel_numbers  = rp.linspace(1,n_cw,n_cw, dtype=rp.int16)
        
        leading_edge_indices    = rp.append(leading_edge_indices   , LE_inds       ) 
        trailing_edge_indices   = rp.append(trailing_edge_indices  , TE_inds       )            
        panels_per_strip        = rp.append(panels_per_strip       , RNMAX         )
        chordwise_panel_number  = rp.append(chordwise_panel_number , panel_numbers )       

    # xyz positions for the right side of this fuselage's outermost panels
    fhs_x = fhs_x.at[-(n_cw+1):].set(rp.concatenate([fhs_xi_b1,rp.array([fhs_xi_b2[-1]])])+ fus.origin[0][0])
    fhs_y = fhs_y.at[-(n_cw+1):].set(rp.ones(n_cw+1)*fhs_eta_b[idx_y]  + fus.origin[0][1])
    fhs_z = fhs_z.at[-(n_cw+1):].set(rp.zeros(n_cw+1)                  + fus.origin[0][2])
    fvs_x = fvs_x.at[-(n_cw+1):].set(rp.concatenate([fvs_xi_a1,rp.array([fvs_xi_a2[-1]])]) + fus.origin[0][0])
    fvs_z = fvs_z.at[-(n_cw+1):].set(rp.ones(n_cw+1)*fvs_eta_a[idx_y] + fus.origin[0][2])
    fvs_y = fvs_y.at[-(n_cw+1):].set(rp.zeros(n_cw+1)                 + fus.origin[0][1])
    fhs_cs =  (fhs.chord[:-1]+fhs.chord[1:])/2
    fvs_cs =  (fvs.chord[:-1]+fvs.chord[1:])/2     

    # Horizontal Fuselage Sections 
    wing_areas = []
    wing_areas.append(rp.array(fus_h_area, dtype=precision))
    wing_areas.append(rp.array(fus_h_area, dtype=precision))
    
    # store points of horizontal section of fuselage 
    fhs_cs  = rp.concatenate([fhs_cs, fhs_cs])
    fhs_xah = rp.concatenate([fhs_xah, fhs_xah])
    fhs_yah = rp.concatenate([fhs_yah,-fhs_yah])
    fhs_zah = rp.concatenate([fhs_zah, fhs_zah])
    fhs_xbh = rp.concatenate([fhs_xbh, fhs_xbh])
    fhs_ybh = rp.concatenate([fhs_ybh,-fhs_ybh])
    fhs_zbh = rp.concatenate([fhs_zbh, fhs_zbh])
    fhs_xch = rp.concatenate([fhs_xch, fhs_xch])
    fhs_ych = rp.concatenate([fhs_ych,-fhs_ych])
    fhs_zch = rp.concatenate([fhs_zch, fhs_zch])
    fhs_xa1 = rp.concatenate([fhs_xa1, fhs_xa1])
    fhs_ya1 = rp.concatenate([fhs_ya1,-fhs_ya1])
    fhs_za1 = rp.concatenate([fhs_za1, fhs_za1])
    fhs_xa2 = rp.concatenate([fhs_xa2, fhs_xa2])
    fhs_ya2 = rp.concatenate([fhs_ya2,-fhs_ya2])
    fhs_za2 = rp.concatenate([fhs_za2, fhs_za2])
    fhs_xb1 = rp.concatenate([fhs_xb1, fhs_xb1])
    fhs_yb1 = rp.concatenate([fhs_yb1,-fhs_yb1])    
    fhs_zb1 = rp.concatenate([fhs_zb1, fhs_zb1])
    fhs_xb2 = rp.concatenate([fhs_xb2, fhs_xb2])
    fhs_yb2 = rp.concatenate([fhs_yb2,-fhs_yb2])            
    fhs_zb2 = rp.concatenate([fhs_zb2, fhs_zb2])
    fhs_xac = rp.concatenate([fhs_xac, fhs_xac])
    fhs_yac = rp.concatenate([fhs_yac,-fhs_yac])
    fhs_zac = rp.concatenate([fhs_zac, fhs_zac])            
    fhs_xbc = rp.concatenate([fhs_xbc, fhs_xbc])
    fhs_ybc = rp.concatenate([fhs_ybc,-fhs_ybc])
    fhs_zbc = rp.concatenate([fhs_zbc, fhs_zbc])
    fhs_xc  = rp.concatenate([fhs_xc , fhs_xc ])
    fhs_yc  = rp.concatenate([fhs_yc ,-fhs_yc])
    fhs_zc  = rp.concatenate([fhs_zc , fhs_zc ])     
    fhs_x   = rp.concatenate([fhs_x  , fhs_x  ])
    fhs_y   = rp.concatenate([fhs_y  ,-fhs_y ])
    fhs_z   = rp.concatenate([fhs_z  , fhs_z  ])      
    
    if model_geometry == True:
        
        # increment fuslage lifting surface sections  
        VD.n_fus   += 2    
        VD.n_cp    += len(fhs_xch)
        VD.n_w     += 2 
        VD.counter += 1
        
        # store this fuselage's discretization information 
        n_panels         = n_sw*n_cw
        first_panel_ind  = VD.XAH.size
        first_strip_ind  = [VD.chordwise_breaks.size, VD.chordwise_breaks.size+n_sw]
        chordwise_breaks =  first_panel_ind + rp.arange(0,2*n_panels)[0::n_cw]        
        
        VD.chordwise_breaks = rp.append(VD.chordwise_breaks, chordwise_breaks)
        VD.spanwise_breaks  = rp.append(VD.spanwise_breaks , first_strip_ind )           
        VD.n_sw             = rp.append(VD.n_sw            , [n_sw, n_sw]    )
        VD.n_cw             = rp.append(VD.n_cw            , [n_cw, n_cw]    )
        VD.surface_ID       = rp.append(VD.surface_ID      , rp.ones(len(fhs_xch)) * VD.counter)
        VD.surface_ID_full  = rp.append(VD.surface_ID_full , rp.ones(((n_sw+1)*(n_cw+1))*2) * VD.counter)
        
        VD.leading_edge_indices      = rp.append(VD.leading_edge_indices     , rp.tile(leading_edge_indices        , 2) )
        VD.trailing_edge_indices     = rp.append(VD.trailing_edge_indices    , rp.tile(trailing_edge_indices       , 2) )           
        VD.panels_per_strip          = rp.append(VD.panels_per_strip         , rp.tile(panels_per_strip            , 2) )
        VD.chordwise_panel_number    = rp.append(VD.chordwise_panel_number   , rp.tile(chordwise_panel_number      , 2) ) 
        VD.exposed_leading_edge_flag = rp.append(VD.exposed_leading_edge_flag, rp.tile(rp.ones(n_sw,dtype=rp.int16), 2) )
        
        # Store fus in vehicle vector  
        VD.XAH    = rp.append(VD.XAH  , rp.array(fhs_xah  , dtype=precision))
        VD.YAH    = rp.append(VD.YAH  , rp.array(fhs_yah  , dtype=precision))
        VD.ZAH    = rp.append(VD.ZAH  , rp.array(fhs_zah  , dtype=precision))
        VD.XBH    = rp.append(VD.XBH  , rp.array(fhs_xbh  , dtype=precision))
        VD.YBH    = rp.append(VD.YBH  , rp.array(fhs_ybh  , dtype=precision))
        VD.ZBH    = rp.append(VD.ZBH  , rp.array(fhs_zbh  , dtype=precision))
        VD.XCH    = rp.append(VD.XCH  , rp.array(fhs_xch  , dtype=precision))
        VD.YCH    = rp.append(VD.YCH  , rp.array(fhs_ych  , dtype=precision))
        VD.ZCH    = rp.append(VD.ZCH  , rp.array(fhs_zch  , dtype=precision))
        VD.XA1    = rp.append(VD.XA1  , rp.array(fhs_xa1  , dtype=precision))
        VD.YA1    = rp.append(VD.YA1  , rp.array(fhs_ya1  , dtype=precision))
        VD.ZA1    = rp.append(VD.ZA1  , rp.array(fhs_za1  , dtype=precision))
        VD.XA2    = rp.append(VD.XA2  , rp.array(fhs_xa2  , dtype=precision))
        VD.YA2    = rp.append(VD.YA2  , rp.array(fhs_ya2  , dtype=precision))
        VD.ZA2    = rp.append(VD.ZA2  , rp.array(fhs_za2  , dtype=precision))
        VD.XB1    = rp.append(VD.XB1  , rp.array(fhs_xb1  , dtype=precision))
        VD.YB1    = rp.append(VD.YB1  , rp.array(fhs_yb1  , dtype=precision))
        VD.ZB1    = rp.append(VD.ZB1  , rp.array(fhs_zb1  , dtype=precision))
        VD.XB2    = rp.append(VD.XB2  , rp.array(fhs_xb2  , dtype=precision))
        VD.YB2    = rp.append(VD.YB2  , rp.array(fhs_yb2  , dtype=precision))
        VD.ZB2    = rp.append(VD.ZB2  , rp.array(fhs_zb2  , dtype=precision))
        VD.XAC    = rp.append(VD.XAC  , rp.array(fhs_xac  , dtype=precision))
        VD.YAC    = rp.append(VD.YAC  , rp.array(fhs_yac  , dtype=precision))
        VD.ZAC    = rp.append(VD.ZAC  , rp.array(fhs_zac  , dtype=precision))
        VD.XBC    = rp.append(VD.XBC  , rp.array(fhs_xbc  , dtype=precision))
        VD.YBC    = rp.append(VD.YBC  , rp.array(fhs_ybc  , dtype=precision))
        VD.ZBC    = rp.append(VD.ZBC  , rp.array(fhs_zbc  , dtype=precision))
        VD.XC     = rp.append(VD.XC   , rp.array(fhs_xc   , dtype=precision))
        VD.YC     = rp.append(VD.YC   , rp.array(fhs_yc   , dtype=precision))
        VD.ZC     = rp.append(VD.ZC   , rp.array(fhs_zc   , dtype=precision))
        VD.CS     = rp.append(VD.CS   , rp.array(fhs_cs   , dtype=precision))
        VD.X      = rp.append(VD.X    , rp.array(fhs_x    , dtype=precision))
        VD.Y      = rp.append(VD.Y    , rp.array(fhs_y    , dtype=precision))
        VD.Z      = rp.append(VD.Z    , rp.array(fhs_z    , dtype=precision)) 
        VD.wing_areas = rp.append(VD.wing_areas, rp.array(wing_areas))
        
        VL = VD.vortex_lift
        VL.append(False)
        VL.append(False)
    
    
    return VD
