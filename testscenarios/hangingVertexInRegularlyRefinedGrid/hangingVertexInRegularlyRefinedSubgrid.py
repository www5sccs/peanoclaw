#!/usr/bin/env python
# encoding: utf-8
#This scenario constructs a situation where grid coarsening is
#triggered in such a way that logically a hanging vertex would
#be created that is surrounded by four refined cells, since
#adjacent vertices an both sides are refined.

from __future__ import division

"""
2D shallow water equations.
"""
#===========================================================================
# Import libraries
#===========================================================================

import numpy as np

def qinit(state,hl,ul,vl,hr,ur,vr,radDam):
    import math
    x0=0.5
    xCenter = state.grid.x.centers
    yCenter = state.grid.y.centers
    
    Y,X = np.meshgrid(yCenter,xCenter)
    r = abs(X-x0)
    state.q[0,:,:] = hl*(r<=radDam) + hr*(r>radDam)
    state.q[1,:,:] = 0#hl*ul*(r<=radDam) + hr*ur*(r>radDam)
    state.q[2,:,:] = 0#hl*vl*(r<=radDam) + hr*vr*(r>radDam)
    
def refinement_criterion_time_dependent(state):
    center_x = 0.5 - 1.0/18.0
    distance_of_front = 1.9/18.0 + state.t * 100.0 / 180.0
    
    import math
    dimension_x = state.patch.dimensions[0]
    dimension_y = state.patch.dimensions[1]
    
    cell_position_x = (dimension_x.lower + dimension_x.upper) / 2.0
    cell_size_x = (dimension_x.upper - dimension_x.lower)
    minimal_distance_to_front = min(abs(center_x + distance_of_front - cell_position_x) \
    							,abs(center_x - distance_of_front - cell_position_x))
    
    if (minimal_distance_to_front <= cell_size_x / 2.0):
        return 1.0/(6.0*27.0)
    elif (minimal_distance_to_front >= cell_size_x):
        return 1.0/(6.0*9.0)
    else:
        return dimension_x.delta


    
def shallow2D(use_petsc=False,iplot=0,htmlplot=False,outdir='./_output',solver_type='classic',amr_type=None):
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
    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D()
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split=1
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D()

    from clawpack import riemann
    solver.rp = riemann.rp2_shallow_roe_with_efix
    solver.num_waves = 3

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.wall

    #===========================================================================
    # Initialize domain and state, then initialize the solution associated to the 
    # state and finally initialize aux array
    #===========================================================================

    # resolution of each grid
    mgrid = 6

    # number of initial AMR grids in each dimension
    msubgrid = 9

    if amr_type is not None:
        m = mgrid
    else:
        # number of Domain grid cells expressed as the product of
        # grid resolution and the number of AMR sub-grids for
        # easy comparison between the two methods
        m = mgrid*msubgrid
    
    mx = m
    my = m

    # Domain:
    xlower = 0
    xupper = 1
    ylower = 0
    yupper = 1

    x = pyclaw.Dimension('x',xlower,xupper,mx)
    y = pyclaw.Dimension('y',ylower,yupper,my)
    domain = pyclaw.Domain([x,y])

    num_eqn = 3  # Number of equations
    state = pyclaw.State(domain,num_eqn)

    grav = 1.0 # Parameter (global auxiliary variable)
    state.problem_data['grav'] = grav

    # Initial solution
    # ================
    # Riemann states of the dam break problem
    damRadius = 0.25 #0.5
    hl = 2.
    ul = 0.
    vl = 0.
    hr = 1.
    ur = 0.
    vr = 0.

    qinit(state,hl,ul,vl,hr,ur,vl,damRadius) # This function is defined above

    # this closure is used by AMR-style codes
    def qinit_callback(state):
        qinit(state,hl,ul,vl,hr,ur,vl,damRadius)

    #===========================================================================
    # Set up controller and controller parameters
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal = 0.1 #0.03

    if amr_type is not None:        
        if amr_type == 'peano':
            import clawpack.peanoclaw as amrclaw
            claw.solver = amrclaw.Solver(solver
                                        ,1/(mgrid*msubgrid)
                                        ,qinit_callback
                                        ,refinement_criterion=refinement_criterion_time_dependent
                                        )
            claw.solution = amrclaw.Solution(state, domain)
        else:
            raise Exception('unsupported amr_type %s' % amr_type)
    else:
        claw.solver = solver
        claw.solution = pyclaw.Solution(state,domain)
        
    #claw.keep_copy = True
    #claw.outdir = outdir
 
    claw.keep_copy = False
    claw.output_format = None
    claw.outdir = None

    claw.num_output_times = 10

    #===========================================================================
    # Solve the problem
    #===========================================================================

    claw.run_prepare()
    print 'running on rank: ', claw.solver.peano.rank
    if claw.solver.peano.rank == 0:
        status = claw.runMaster()
    else:
        status = claw.solver.peano.runWorker()

    #===========================================================================
    # Plot results
    #===========================================================================
    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

    return claw

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(shallow2D)
