#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import clawpack.peanoclaw as peanoclaw

def init(state):
    zr = 2.0  # Impedance in right half
    cr = 2.0  # Sound speed in right half
    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    state.aux[0,:,:,:] = zl*(X<0.) + zr*(X>=0.) # Impedance
    state.aux[1,:,:,:] = cl*(X<0.) + cr*(X>=0.) # Sound speed

    x0 = 0.; y0 = 0.; z0 = 0.
    r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    width=0.1
    state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))

    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.
 
def acoustics3D(iplot=False,htmlplot=False,use_petsc=False,outdir='./_output',solver_type='classic',**kwargs):
    """
    Example python script for solving the 3d acoustics equations.
    """
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
    subdivisionFactor = 6
    if solver_type=='classic':
        solver=pyclaw.ClawSolver3D()
    else:
        raise Exception('Unrecognized solver_type.')

    from clawpack import riemann

    # Peano Solver
    peanoSolver = peanoclaw.Solver(solver, (2./3.)/subdivisionFactor, init)

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
    # solver.bc_lower[0]    =pyclaw.BC.wall
    # solver.bc_lower[1]    =pyclaw.BC.wall
    # solver.bc_lower[2]    =pyclaw.BC.wall
    # solver.aux_bc_lower[0]=pyclaw.BC.wall
    # solver.aux_bc_lower[1]=pyclaw.BC.wall
    # solver.aux_bc_lower[2]=pyclaw.BC.wall
    mx=30; my=30; mz=30

    #===========================================================================
    # Initialize domain and state, then initialize the solution associated to the 
    # state and finally initialize aux array
    #===========================================================================

    # Initialize domain
    mx = subdivisionFactor
    my = subdivisionFactor
    mz = subdivisionFactor
    x = pyclaw.Dimension('x',-1.0,1.0,mx)
    y = pyclaw.Dimension('y',-1.0,1.0,my)
    z = pyclaw.Dimension('z',-1.0,1.0,mz)
    domain = pyclaw.Domain([x,y,z])

    num_eqn = 4
    num_aux = 2 # density, sound speed
    state = pyclaw.State(domain,num_eqn,num_aux)

    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal = 5.0
    claw.keep_copy = True
    claw.solution = peanoclaw.solution.Solution(state,domain) #pyclaw.Solution(state,domain)
    claw.solver = peanoSolver  #solver
    claw.outdir=outdir

    #===========================================================================
    # Solve the problem
    #===========================================================================
    status = claw.run()

    #===========================================================================
    # Plot results
    #===========================================================================
    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir,file_format=claw.output_format)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir,file_format=claw.output_format)

if __name__=="__main__":
    import sys
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(acoustics3D)
