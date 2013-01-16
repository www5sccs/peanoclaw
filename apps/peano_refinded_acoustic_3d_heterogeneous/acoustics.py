#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import clawpack.peanoclaw as peanoclaw
import benchmark as BM
#from pylab import *

XI = -1.0
XF = 1.0
LENGTH = XF - XI
SUBDIVISION_FACTOR = 8
CELLS = 2
INIT_MIN_MESH_WIDTH = LENGTH / CELLS / SUBDIVISION_FACTOR
NUM_OUTPUT_TIMES = 2
TFINAL = 0.0001

# size should be equal to 2^n + 1 to generate the landscape with diamond-square algorithms
# it means that CELLS * SUBDIVISION_FACTOR should be 2^n
SIZE = CELLS * SUBDIVISION_FACTOR + 1  
NUM_LAYERS = 1
LANDSCAPE = BM.generate_landscape_layers( NUM_LAYERS, XI, XF, SIZE)
BM.vis_landscape( LANDSCAPE, XI, XF,  SIZE, NUM_LAYERS)

def init_aux_landscaple(state):
    global XI
    global XF
    global CELLS
    global SUBDIVISION_FACTOR
    global SIZE
    global NUM_LAYERS
    global LANDSCAPE

    ly = LANDSCAPE[0]
    ls = ly[1:,1:]
    
    zr = 1.0  # Impedance in right half
    cr = 1.0  # Sound speed in right half
    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    LU = np.linspace( XI, XF, num=SIZE)
    LV = np.linspace( XI, XF, num=SIZE)
    LX, LY = np.meshgrid(LU, LV)

    LX = LX[1:,1:]
    LY = LY[1:,1:]

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    XMIN = np.amin(X)
    YMIN = np.amin(Y)
    XMAX = np.amax(X)
    YMAX = np.amax(Y)
    print X.shape,ls.shape

    # print ls
    border = 0.35
    state.aux[0,:,:,:] = zl*(X<border) + zr*(X>=border) # Impedance
    state.aux[1,:,:,:] = cl*(X<border) + cr*(X>=border) # Sound speed
    

def init_aux1(state):
    zr = 1.0  # Impedance in right half
    cr = 1.0  # Sound speed in right half
    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    border = 0.35
    state.aux[0,:,:,:] = zl*(X<border) + zr*(X>=border) # Impedance
    state.aux[1,:,:,:] = cl*(X<border) + cr*(X>=border) # Sound speed

def init_field(state):
    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    x0 = -0.5; y0 = 0.0; z0 = 0.0
    r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    width=0.07
    state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))
    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.


def init(state):
    init_aux1(state)
    #init_aux_landscaple(state)
    init_field(state)

def refinement_criterion(state):
    global INIT_MIN_MESH_WIDTH
    if state.aux[0,:,:,:].max() > 1:
        return INIT_MIN_MESH_WIDTH / 2.0
    return INIT_MIN_MESH_WIDTH
 
def acoustics3D(iplot=False,htmlplot=False,use_petsc=False,outdir='./_output',solver_type='classic',**kwargs):
    """
    Example python script for solving the 3d acoustics equations.
    """
    #===========================================================================
    # Global variables
    #===========================================================================
    global INIT_MIN_MESH_WIDTH
    global SUBDIVISION_FACTOR
    global XI
    global XF
    global TFINAL
    global NUM_OUTPUT_TIMES

    #===========================================================================
    # Import libraries
    #===========================================================================
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    #===========================================================================
    # Setup solver and solver parameters
    #===========================================================================
    if solver_type=='classic':
        solver=pyclaw.ClawSolver3D()
    else:
        raise Exception('Unrecognized solver_type.')

    from clawpack import riemann

    # Peano Solver
    peanoSolver = peanoclaw.Solver(solver, 
                                   INIT_MIN_MESH_WIDTH,
                                   init, 
                                   refinement_criterion=refinement_criterion
                                   )

    solver.rp = riemann.rp3_vc_acoustics
    solver.num_waves = 2
    solver.limiters = pyclaw.limiters.tvd.MC

    solver.bc_lower[0]=pyclaw.BC.extrap
    solver.bc_upper[0]=pyclaw.BC.extrap
    solver.bc_lower[1]=pyclaw.BC.extrap
    solver.bc_upper[1]=pyclaw.BC.extrap
    solver.bc_lower[2]=pyclaw.BC.extrap
    solver.bc_upper[2]=pyclaw.BC.extrap

    solver.aux_bc_lower[0]=pyclaw.BC.extrap
    solver.aux_bc_upper[0]=pyclaw.BC.extrap
    solver.aux_bc_lower[1]=pyclaw.BC.extrap
    solver.aux_bc_upper[1]=pyclaw.BC.extrap
    solver.aux_bc_lower[2]=pyclaw.BC.extrap
    solver.aux_bc_upper[2]=pyclaw.BC.extrap

    solver.dimensional_split=True
    solver.limiters = pyclaw.limiters.tvd.MC

    #===========================================================================
    # Initialize domain and state, then initialize the solution associated to the 
    # state and finally initialize aux array
    #===========================================================================

    # Initialize domain
    mx = SUBDIVISION_FACTOR
    my = SUBDIVISION_FACTOR
    mz = SUBDIVISION_FACTOR
    x = pyclaw.Dimension('x', XI, XF, mx)
    y = pyclaw.Dimension('y', XI, XF, my)
    z = pyclaw.Dimension('z', XI, XF, mz)
    domain = pyclaw.Domain([x,y,z])

    num_eqn = 4
    num_aux = 2 # density, sound speed
    state = pyclaw.State(domain,num_eqn,num_aux)

    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal = TFINAL
    claw.keep_copy = True
    claw.solution = peanoclaw.solution.Solution(state,domain) #pyclaw.Solution(state,domain)
    claw.solver = peanoSolver  #solver
    claw.outdir=outdir
    claw.num_output_times = NUM_OUTPUT_TIMES

    #===========================================================================
    # Solve the problem
    #===========================================================================
    status = claw.run()

if __name__=="__main__":
    import sys
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(acoustics3D)
