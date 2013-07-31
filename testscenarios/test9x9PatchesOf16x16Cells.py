#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

"""
2D shallow water equations.
"""
#===========================================================================
# Import libraries
#===========================================================================

import numpy as np

def qinit(state,hl,ul,vl,hr,ur,vr,radDam):
    x0=0.5
    y0=0.5
    xCenter = state.grid.x.centers
    yCenter = state.grid.y.centers
    
    Y,X = np.meshgrid(yCenter,xCenter)
    r = np.sqrt((X-x0)**2 + (Y-y0)**2)
    state.q[0,:,:] = hl*(r<=radDam) + hr*(r>radDam)
    state.q[1,:,:] = hl*ul*(r<=radDam) + hr*ur*(r>radDam)
    state.q[2,:,:] = hl*vl*(r<=radDam) + hr*vr*(r>radDam)
    
#     for x in xrange(6):
#       for y in xrange(6):
#         state.q[0, x, y] = x + y*100 + 1
    
def refinement_criterion_tmp(state):
    if((state.q[0,:,:].max() - state.q[0,:,:].min()) > 0.2):
        return 1.0/(4.0*81.0)
    else:
        return 1.0/(6.0*9.0)
      
def refinement_criterion(state):
    center_x = 0.5
    center_y = 0.5
    radius1 = 0.25
    
    import math
    dimension_x = state.patch.dimensions[0]
    dimension_y = state.patch.dimensions[1]
    
    distance_to_circle1 = abs(math.sqrt(((dimension_x.lower + dimension_x.upper) / 2 - center_x) ** 2 
                          + ((dimension_y.lower + dimension_y.upper) / 2 - center_y) ** 2) - radius1)
    
    if (distance_to_circle1 < (dimension_x.upper - dimension_x.lower) / 2):
        return 1.0/(6.0*81.0)
        #return 1.0/(6.0*27.0)
    elif (distance_to_circle1 > (dimension_x.upper - dimension_x.lower) * 1.5 
        #  and distance_to_circle2 > (dimension_x.upper - dimension_x.lower) * 1.5
        #  and distance_to_circle3 > (dimension_x.upper - dimension_x.lower) * 1.5
        ):
        return 1.0/(6.0*9.0)
    else:
        return dimension_x.delta
    
def refinement_criterion_time_dependent(state):
    center_x = 0.5
    center_y = 0.5
    radius1 = 0.25 + state.t * 0.75 / 0.5
    radius2 = 0.25 - state.t * 0.5 / 0.5
    radius3 = 0.25 - state.t * 0.7 / 0.5
    
    import math
    dimension_x = state.patch.dimensions[0]
    dimension_y = state.patch.dimensions[1]
    
    distance_to_circle1 = abs(math.sqrt(((dimension_x.lower + dimension_x.upper) / 2 - center_x) ** 2 
                          + ((dimension_y.lower + dimension_y.upper) / 2 - center_y) ** 2) - radius1)
    distance_to_circle2 = abs(math.sqrt(((dimension_x.lower + dimension_x.upper) / 2 - center_x) ** 2 
                          + ((dimension_y.lower + dimension_y.upper) / 2 - center_y) ** 2) - radius2)
    distance_to_circle3 = abs(math.sqrt(((dimension_x.lower + dimension_x.upper) / 2 - center_x) ** 2 
                          + ((dimension_y.lower + dimension_y.upper) / 2 - center_y) ** 2) - radius3)
    
    if (distance_to_circle1 < (dimension_x.upper - dimension_x.lower) / 2
        or distance_to_circle2 < (dimension_x.upper - dimension_x.lower) / 2
        or distance_to_circle3 < (dimension_x.upper - dimension_x.lower) / 2
        ):
        return 1.0/(6.0*27.0)
    elif (distance_to_circle1 > (dimension_x.upper - dimension_x.lower) * 1.5 
        and distance_to_circle2 > (dimension_x.upper - dimension_x.lower) * 1.5
        and distance_to_circle3 > (dimension_x.upper - dimension_x.lower) * 1.5
        ):
        return 1.0/(6.0*9.0)
    else:
        return dimension_x.delta


    
def shallow2D(use_petsc=False,iplot=0,htmlplot=False,outdir='./_output',solver_type='classic',amr_type="peano"):
    #===========================================================================
    # Import libraries
    #===========================================================================
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    #===========================================================================
    # Setup solver and solver parameters
    #===========================================================================
    if solver_type == 'classic':
        solver = pyclaw.ClawSolver2D(riemann.shallow_roe_with_efix_2D)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.dimensional_split=1
    elif solver_type == 'sharpclaw':
        solver = pyclaw.SharpClawSolver2D(riemann.shallow_roe_with_efix_2D)

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.wall
    solver.bc_lower[1] = pyclaw.BC.wall
    solver.bc_upper[1] = pyclaw.BC.wall

    #===========================================================================
    # Initialize domain and state, then initialize the solution associated to the 
    # state and finally initialize aux array
    #===========================================================================

    # resolution of each grid
    mgrid = 16

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

    state = pyclaw.State(domain,solver.num_eqn)

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
    claw.tfinal = 1e-1 #3e-1 #0.03

    if amr_type is not None:        
        if amr_type == 'peano':
            import peanoclaw as amrclaw
            claw.solver = amrclaw.Solver(solver
                                        ,1/(mgrid*msubgrid)
                                        ,qinit_callback
                                        #,refinement_criterion=refinement_criterion_time_dependent
                                        #,refinement_criterion=refinement_criterion
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
    # Plot results
    #===========================================================================
    if htmlplot:  pyclaw.plot.html_plot(outdir=outdir)
    if iplot:     pyclaw.plot.interactive_plot(outdir=outdir)

    return claw

if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main

    run_app_from_main(shallow2D)        
