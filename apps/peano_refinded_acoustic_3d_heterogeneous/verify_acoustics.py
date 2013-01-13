#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import clawpack.peanoclaw as peanoclaw
import wavefront as wf
from clawpack import pyclaw

OBJECTS = None
XI = -1.0
XF = 1.0
LENGTH = XF - XI
SUBDIVISION_FACTOR = 6
CELLS = 3
INIT_MIN_MESH_WIDTH = LENGTH / CELLS / SUBDIVISION_FACTOR
NUM_OUTPUT_TIMES = 10
TEST_LEVEL = 5
NUM_EQS = 4
TFINAL = 1.0

def init_objects(filename):
    global OBJECTS
    OBJECTS = wf.read_obj(filename)

def init_aux(state):
    global OBJECTS
    zr = 1.0  # Impedance in right half
    cr = 1.0  # Sound speed in right half

    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half
    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    if OBJECTS is not None:
        vertices = OBJECTS[0].vertices
        vert_x = [vert[0] for vert in vertices ]
        vert_y = [vert[1] for vert in vertices ]
        vert_z = [vert[2] for vert in vertices ]
        XMAX = max(vert_x)
        XMIN = min(vert_x)
        YMAX = max(vert_y)
        YMIN = min(vert_y)
        ZMAX = max(vert_z)
        ZMIN = min(vert_z)
    
        # print 'XMIN',  XMIN
        # print 'XMAX',  XMAX
        inside_the_object = np.logical_and(X > XMIN , X < XMAX )#, Y > YMIN, Y < YMAX, Z > ZMIN, Z < ZMAX)
        state.aux[0,:,:,:] = zl*(np.logical_not(inside_the_object)) + zr*(inside_the_object) # Impedance
        state.aux[1,:,:,:] = cl*(np.logical_not(inside_the_object)) + cr*(inside_the_object) # Sound speed
    else:
        border = 0.0#0.35
        state.aux[0,:,:,:] = zl*(X<border) + zr*(X>=border) # Impedance
        state.aux[1,:,:,:] = cl*(X<border) + cr*(X>=border) # Sound speed

def init_q(state):
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
    init_aux(state)
    init_q(state)

def refinement_criterion(state):
    global SUBDIVISION_FACTOR
    global XI
    global XF
    global LENTGH
    global CELLS
    global INIT_MIN_MESH_WIDTH
    # grid = state.grid;
    # num_dim = grid.num_dim
    # delta = grid.delta
    # xwidth = delta[0]
    if state.aux[0,:,:,:].max() > 1:
        return INIT_MIN_MESH_WIDTH / 2.0
 
    return INIT_MIN_MESH_WIDTH
 
def acoustics3D(iplot=False,htmlplot=False,use_petsc=False,outdir='./_output',solver_type='classic',**kwargs):
    """
    Example python script for solving the 3d acoustics equations.
    """
    #===========================================================================
    # Import libraries
    #===========================================================================
    global INIT_MIN_MESH_WIDTH
    global NUM_OUTPUT_TIMES
    global TFINAL
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

    #solver.before_step = _probe
    status = claw.run()

    return claw

def pyclaw_acoustics3D(iplot=False,htmlplot=False,use_petsc=False,outdir='./_output',solver_type='classic',**kwargs):
    """
    Example python script for solving the 3d acoustics equations.
    """
    global NUM_OUTPUT_TIMES
    global TFINAL
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        solver=pyclaw.ClawSolver3D()
    else:
        raise Exception('Unrecognized solver_type.')

    from clawpack import riemann
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
    mx = SUBDIVISION_FACTOR * CELLS
    my = SUBDIVISION_FACTOR * CELLS
    mz = SUBDIVISION_FACTOR * CELLS
    x = pyclaw.Dimension('x', XI, XF, mx)
    y = pyclaw.Dimension('y', XI, XF, my)
    z = pyclaw.Dimension('z', XI, XF, mz)
    domain = pyclaw.Domain([x,y,z])

    num_eqn = 4
    num_aux = 2 # density, sound speed
    state = pyclaw.State(domain,num_eqn,num_aux)
    init(state)

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir=outdir
    claw.num_output_times = NUM_OUTPUT_TIMES

    # Solve
    claw.tfinal = TFINAL
    status = claw.run()
    return claw

def verify(claw1, claw2):
    global NUM_OUTPUT_TIMES
    global TEST_LEVEL
    global NUM_EQS
    global CELLS
    global SUBDIVISION_FACTOR

    import assembleState as AS
    import matplotlib.pyplot as plt

    for i in range(NUM_OUTPUT_TIMES):
        q_peano = AS.assemblePeanoClawState(claw1.frames[i], claw2.frames[i].domain, NUM_EQS, CELLS * SUBDIVISION_FACTOR)
        res1 = q_peano[:,:,TEST_LEVEL]
        res2 = claw2.frames[i].state.q[0,:,:,TEST_LEVEL]
        plt.pcolor(res2 - res1)
        plt.colorbar()
        plt.show()
            


if __name__=="__main__":
    import sys
    from clawpack.pyclaw.util import run_app_from_main
    #init_objects('cube.obj')
    pyclaw_output = run_app_from_main(pyclaw_acoustics3D)
    peanoclaw_output = run_app_from_main(acoustics3D)
    verify( peanoclaw_output, pyclaw_output)
