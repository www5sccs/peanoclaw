#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import clawpack.peanoclaw as peanoclaw
import benchmark as BM
#from pylab import *

XI = -1.0
XF = 1.0
LENGTH = XF - XI
SUBDIVISION_FACTOR = 16
CELLS = 3
INIT_MIN_MESH_WIDTH = (LENGTH / CELLS) / SUBDIVISION_FACTOR
NUM_OUTPUT_TIMES = 30
TFINAL = 2.0#0001

# size should be equal to 2^n + 1 to generate the landscape with diamond-square algorithms
# it means that CELLS * SUBDIVISION_FACTOR should be 2^n
NUM_GRID_POINTS = CELLS * SUBDIVISION_FACTOR + 1  
NUM_LAYERS = 3
LAYERS_AUX_DATA = [0.2,0.5,0.2,0.5] #np.arange(NUM_LAYERS+1)#

print "Generating landscape! please be patient!"
LANDSCAPE = BM.generate_landscape_layers( NUM_LAYERS, XI, XF, NUM_GRID_POINTS)
LS_SIZE =  len(LANDSCAPE[0])
BM.vis_landscape( LANDSCAPE, XI, XF, LS_SIZE, NUM_LAYERS)

#cutting the landscape to the appropriate size
SIZE = NUM_GRID_POINTS - 1
RESIZED_LS = []
for ls in LANDSCAPE:
    ls = ls[:SIZE,:SIZE]
    #print ls.shape
    RESIZED_LS.append(ls)
BM.vis_landscape( RESIZED_LS, XI, XF, SIZE, NUM_LAYERS)

GLOBAL_AUX = np.zeros((SIZE,SIZE,SIZE))
idx_to_length_map =  np.linspace( XI, XF, num=SIZE)

for xidx in range(SIZE):
    for yidx in range(SIZE):
        for zidx in range(SIZE):
            myheight = idx_to_length_map[zidx]
            layers_heights = np.zeros((NUM_LAYERS))
            for layer in range(NUM_LAYERS):
                layers_heights[layer] = RESIZED_LS[layer][xidx,yidx]

            bool_array = layers_heights < myheight
            #print bool_array
            found = np.where(bool_array == False )[0]
            if len( found ) is 0:
                layer_idx = NUM_LAYERS
            else:
                layer_idx = found[0]
            #print layer_idx
            GLOBAL_AUX[xidx,yidx,zidx] = LAYERS_AUX_DATA[layer_idx] #aux_val

def init_aux_landscaple(state):
    global XI
    global XF
    global CELLS
    global SUBDIVISION_FACTOR
    global SIZE
    global NUM_LAYERS
    global GLOBAL_AUX

    idx_to_length_map =  np.linspace( XI, XF, num=CELLS+1)
    #print idx_to_length_map

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    XMIN = np.amin(X)
    YMIN = np.amin(Y)
    ZMIN = np.amin(Z)
    XMAX = np.amax(X)
    YMAX = np.amax(Y)
    ZMAX = np.amax(Z)
    XAVG = ( XMIN + XMAX ) / 2.0
    YAVG = ( YMIN + YMAX ) / 2.0
    ZAVG = ( ZMIN + ZMAX ) / 2.0
    xbool = idx_to_length_map < XAVG
    ybool = idx_to_length_map < YAVG
    zbool = idx_to_length_map < ZAVG
    xfound = np.where(xbool == False )[0]
    yfound = np.where(ybool == False )[0]
    zfound = np.where(zbool == False )[0]
    cxindex = xfound[0] - 1
    cyindex = yfound[0] - 1
    czindex = zfound[0] - 1

    # print idx_to_length_map
    # print XAVG
    # print xbool
    # print xfound
    # print cxindex

    cell_index = [cxindex, cyindex, czindex]
    print 'CELL INDEX:',cell_index

    local_aux = GLOBAL_AUX[ cxindex*SUBDIVISION_FACTOR:(cxindex+1)*SUBDIVISION_FACTOR, \
                       cyindex*SUBDIVISION_FACTOR:(cyindex+1)*SUBDIVISION_FACTOR, \
                       czindex*SUBDIVISION_FACTOR:(czindex+1)*SUBDIVISION_FACTOR  \
                           ]
    state.aux[0,:,:,:] = local_aux
    state.aux[1,:,:,:] = 1#local_aux
    

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

    x0 = -0.5; y0 = 0.0; z0 = -0.5
    r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    width=0.3
    state.q[0,:,:,:] = 5*(np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))
    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.


def init(state):
    #init_aux1(state)
    init_aux_landscaple(state)
    init_field(state)

def refinement_criterion(state):
    global INIT_MIN_MESH_WIDTH
    # if state.aux[0,:,:,:].max() > 1:
    #     return INIT_MIN_MESH_WIDTH / 2.0
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
