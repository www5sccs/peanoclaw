'''
Created on Mar 17, 2012

@author: kristof
'''
import clawpack.pyclaw as pyclaw
from ctypes import CFUNCTYPE
from ctypes import py_object
from ctypes import c_int
from ctypes import c_double
from ctypes import c_void_p

from DimensionConverter import get_dimension

class Solution(pyclaw.solution.Solution):
    r"""
    This Solution class is just an extension of the normal pyclaw.Solution. It offers functionality for writing solutions
    when running PyClaw together with Peano for adaptive mesh refinement.
    
    This class is instantiated like the normal Solution, just from the peanoclaw package:
    
        >>> import pyclaw
        >>> x = pyclaw.Dimension('x',0.,1.,100)
        >>> y = pyclaw.Dimension('y',0.,1.,100)
        >>> domain = pyclaw.Domain((x, y))
        >>> state = pyclaw.State(domain,3,2)
        >>> import peanoclaw
        >>> solution = peanoclaw.Solution(state,domain)
    """
    
    CALLBACK_ADD_PATCH_TO_SOLUTION = CFUNCTYPE(None, 
                                                py_object, #q
                                                py_object, #qbc
                                                c_int,     #ghostlayer width
                                                c_double, c_double, c_double, #size
                                                c_double, c_double, c_double, #position
                                                c_double)  #current time
    
    def __init__(self,*arg,**kargs):
        pyclaw.Solution.__init__(self,*arg,**kargs)
        self.peano = None
        self.libpeano = None
        
    def get_add_to_solution_callback(self):
        r"""
        Creates a closure for the callback method to add a grid to the solution.
        """
        def callback_add_to_solution(q, qbc, ghostlayer_width, size_x, size_y, size_z, position_x, position_y, position_z, currentTime):
            dim = get_dimension( q )

            #TODO 3D: Adjust subdivision_factor to 3D
            # Set up grid information for current patch
            if dim is 2:
                subdivision_factor_x = q.shape[1]
                subdivision_factor_y = q.shape[2]
                unknowns_per_subcell = q.shape[0]
                dim_x = pyclaw.Dimension('x', position_x, position_x + size_x, subdivision_factor_x)
                dim_y = pyclaw.Dimension('y', position_y, position_y + size_y, subdivision_factor_y)

                patch = pyclaw.geometry.Patch((dim_x, dim_y))

            elif dim is 3:
                subdivision_factor_x = q.shape[1]
                subdivision_factor_y = q.shape[2]
                subdivision_factor_z = q.shape[3]
                unknowns_per_subcell = q.shape[0]
                dim_x = pyclaw.Dimension('x', position_x, position_x + size_x, subdivision_factor_x)
                dim_y = pyclaw.Dimension('y', position_y, position_y + size_y, subdivision_factor_y)
                dim_z = pyclaw.Dimension('z', position_z, position_z + size_z, subdivision_factor_z)
                
                patch = pyclaw.geometry.Patch((dim_x, dim_y,dim_z))

            state = pyclaw.State(patch, unknowns_per_subcell)
            state.q[:] = q[:]
            state.t = currentTime
            
            self.gathered_patches.append(patch)
            self.gathered_states.append(state)
            
        return self.CALLBACK_ADD_PATCH_TO_SOLUTION(callback_add_to_solution)

    def __deepcopy__(self,memo={}):
        self.gatherStates()
        var =  pyclaw.Solution.__deepcopy__(self, memo)
        var.gathered_states = self.gathered_states
        var.gathered_patches = self.gathered_patches
        return var
    
    def write(self,frame,path='./',file_format='ascii',file_prefix=None,
              write_aux=False,options={},write_p=False):

        self.gatherStates()
        
        self.solution.write(frame, path, file_format,file_prefix,write_aux,options,write_p)
            
    def gatherStates(self):
        r"""
        Writes gathered solution to file.
        """
        if(self.peano == None or self.libpeano == None):
            raise Exception("self.peano and self.libpeano have to be initialized with the reference to the Peano grid by peanoclaw.Solver.setup(...)")

        #Gather patches and states        
        self.gathered_patches = []
        self.gathered_states = []
        
        self.libpeano.pyclaw_peano_gatherSolution.argtypes = [c_void_p, c_void_p]
        self.libpeano.pyclaw_peano_gatherSolution(self.peano, self.get_add_to_solution_callback())
        
        #Assemble solution and write file
        self.domain = pyclaw.Domain(self.gathered_patches)
        self.solution = pyclaw.Solution(self.gathered_states, self.domain)
        
        self.t = self.solution.t      
            
            
