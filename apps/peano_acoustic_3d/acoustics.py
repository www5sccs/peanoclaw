#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import clawpack.peanoclaw as peanoclaw

def init(state):
    # Initial solution
    # ================
    # Riemann states of the dam break problem
    zr = 1.0  # Impedance in right half
    cr = 1.0  # Sound speed in right half
	
    zl = 1.0  # Impedance in left half
    cl = 1.0  # Sound speed in left half

    grid = state.grid
    grid.compute_c_centers()
    X,Y,Z = grid._c_centers

    state.aux[0,:,:,:] = zl*(X<0.) + zr*(X>=0.) # Impedance
    state.aux[1,:,:,:] = cl*(X<0.) + cr*(X>=0.) # Sound speed

    x0 = -0.5; y0 = 0.; z0 = 0.
#    if app == 'test_homogeneous':
    r = np.sqrt((X-x0)**2)
    width=0.2
    state.q[0,:,:,:] = (np.abs(r)<=width)*(1.+np.cos(np.pi*(r)/width))

    # elif app == 'test_heterogeneous' or app == None:
    #     r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    #     width=0.1
    #     state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))

    # else: raise Exception('Unexpected application')
        
    state.q[1,:,:,:] = 0.
    state.q[2,:,:,:] = 0.
    state.q[3,:,:,:] = 0.
    # radDam = 0.2
    #     hl = 2.
    #     ul = 0.
    #     vl = 0.
    #     hr = 1.
    #     ur = 0.
    #     vr = 0.
	
    #     x0=0.5
    #     y0=0.5
    #     xCenter = state.grid.x.centers
    #     yCenter = state.grid.y.centers
	
    #     Y,X = np.meshgrid(yCenter,xCenter)
    #     r = np.sqrt((X-x0)**2 + (Y-y0)**2)
    #     state.q[0,:,:] = hl*(r<=radDam) + hr*(r>radDam)
    #     state.q[1,:,:] = hl*ul*(r<=radDam) + hr*ur*(r>radDam)
    #     state.q[2,:,:] = hl*vl*(r<=radDam) + hr*vr*(r>radDam)


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
    peanoSolver = peanoclaw.Solver(solver, (1./9.)/subdivisionFactor, init)

    solver.rp = riemann.rp3_vc_acoustics
    solver.num_waves = 2
    solver.limiters = pyclaw.limiters.tvd.MC

    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic
    solver.bc_lower[1]=pyclaw.BC.periodic
    solver.bc_upper[1]=pyclaw.BC.periodic
    solver.bc_lower[2]=pyclaw.BC.periodic
    solver.bc_upper[2]=pyclaw.BC.periodic

    solver.aux_bc_lower[0]=pyclaw.BC.periodic
    solver.aux_bc_upper[0]=pyclaw.BC.periodic
    solver.aux_bc_lower[1]=pyclaw.BC.periodic
    solver.aux_bc_upper[1]=pyclaw.BC.periodic
    solver.aux_bc_lower[2]=pyclaw.BC.periodic
    solver.aux_bc_upper[2]=pyclaw.BC.periodic

    # app = None
    # if 'test' in kwargs:
    #     test = kwargs['test']
    #     if test == 'homogeneous':
    #         app = 'test_homogeneous'
    #     elif test == 'heterogeneous':
    #         app = 'test_heterogeneous'
    #     else: raise Exception('Unrecognized test')

#    if app == 'test_homogeneous':
    solver.dimensional_split=True
    mx=256; my=4; mz=4
    zr = 1.0  # Impedance in right half
    cr = 1.0  # Sound speed in right half

    # if app == 'test_heterogeneous' or app == None:
    #     solver.dimensional_split=False
    #     solver.dimensional_split=False
    #     solver.bc_lower[0]    =pyclaw.BC.wall
    #     solver.bc_lower[1]    =pyclaw.BC.wall
    #     solver.bc_lower[2]    =pyclaw.BC.wall
    #     solver.aux_bc_lower[0]=pyclaw.BC.wall
    #     solver.aux_bc_lower[1]=pyclaw.BC.wall
    #     solver.aux_bc_lower[2]=pyclaw.BC.wall
    #     mx=30; my=30; mz=30
    #     zr = 2.0  # Impedance in right half
    #     cr = 2.0  # Sound speed in right half

    solver.limiters = pyclaw.limiters.tvd.MC

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

    # zl = 1.0  # Impedance in left half
    # cl = 1.0  # Sound speed in left half

    # grid = state.grid
    # grid.compute_c_centers()
    # X,Y,Z = grid._c_centers

    # state.aux[0,:,:,:] = zl*(X<0.) + zr*(X>=0.) # Impedance
    # state.aux[1,:,:,:] = cl*(X<0.) + cr*(X>=0.) # Sound speed

    # x0 = -0.5; y0 = 0.; z0 = 0.
    # if app == 'test_homogeneous':
    #     r = np.sqrt((X-x0)**2)
    #     width=0.2
    #     state.q[0,:,:,:] = (np.abs(r)<=width)*(1.+np.cos(np.pi*(r)/width))

    # elif app == 'test_heterogeneous' or app == None:
    #     r = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)
    #     width=0.1
    #     state.q[0,:,:,:] = (np.abs(r-0.3)<=width)*(1.+np.cos(np.pi*(r-0.3)/width))

    # else: raise Exception('Unexpected application')
        
    # state.q[1,:,:,:] = 0.
    # state.q[2,:,:,:] = 0.
    # state.q[3,:,:,:] = 0.

    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal = 2.0
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

    pinitial=claw.frames[0].state.get_q_global()
    pmiddle  =claw.frames[3].state.get_q_global()
    pfinal  =claw.frames[claw.num_output_times].state.get_q_global()

    if pinitial != None and pmiddle != None and pfinal != None:
        pinitial =pinitial[0,:,:,:].reshape(-1)
        pmiddle  =pmiddle[0,:,:,:].reshape(-1)
        pfinal   =pfinal[0,:,:,:].reshape(-1)
        final_difference =np.prod(grid.delta)*np.linalg.norm(pfinal-pinitial,ord=1)
        middle_difference=np.prod(grid.delta)*np.linalg.norm(pmiddle-pinitial,ord=1)

        if app == None:
            print 'Final error: ', final_difference
            print 'Middle error: ', middle_difference

        #import matplotlib.pyplot as plt
        #for i in range(claw.num_output_times):
        #    plt.pcolor(claw.frames[i].state.q[0,:,:,mz/2])
        #    plt.figure()
        #plt.show()

        return pfinal, final_difference

    else:
        
        return

if __name__=="__main__":
    import sys
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(acoustics3D)
