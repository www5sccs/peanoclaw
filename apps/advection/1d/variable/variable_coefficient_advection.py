#!/usr/bin/env python
# encoding: utf-8
"""
_example_with_aux.py

Example python script for solving the 1d variable-coefficient advection 
equation: q_t + u(x)q_x = 0.
"""

import os

import numpy as np



from petclaw.grid import PCDimension as Dimension
from petclaw.grid import PCGrid as Grid
from pyclaw.solution import Solution
from pyclaw.controller import Controller
from petclaw.evolve.petclaw import PetClawSolver1D
from petsc4py import PETSc

def qinit(grid):

    # Initilize petsc Structures for q
    grid.init_q_petsc_structures()
    
    # Initial Data parameters
    ic = grid.aux_global['ic']
    beta = grid.aux_global['beta']
    gamma = grid.aux_global['gamma']
    x0 = grid.aux_global['x0']
    x1 = grid.aux_global['x1']
    x2 = grid.aux_global['x2']

    
    

    # Create an array with fortran native ordering
    
    x =grid.x.center
    
    q=np.zeros([len(x),grid.meqn])
    
    # Gaussian
    qg = np.exp(-beta * (x-x0)**2) * np.cos(gamma * (x - x0))

    # Step Function
    qs = (x > x1) * 1.0 - (x > x2) * 1.0
    
    if ic == 1:
        q[:,0] = qg
    elif ic == 2:
        q[:,0] = qs
    elif ic == 3:
        q[:,0] = qg + qs

    grid.q=q


def auxinit(grid):
    # Initilize petsc Structures for aux
    maux = 1
    xghost=grid.x.centerghost
    #grid.empty_aux(maux)
    grid.aux=np.empty([len(xghost),maux])
    grid.aux[:,0] = np.sin(2.*np.pi*xghost)+2
    #grid.aux= np.reshape(grid.aux, (grid.aux.size, maux))
    
    


# Data paths and objects
example_path = './'
setprob_path = os.path.join(example_path,'setprob.data')

# Initialize grids and solutions
x = Dimension('x',0.0,1.0,100,mthbc_lower=2,mthbc_upper=2,mbc=2)
grid = Grid(x)
grid.set_aux_global(setprob_path)
grid.meqn = 1
grid.t = 0.0
qinit(grid)
auxinit(grid)
init_solution = Solution(grid)

# Solver setup
solver = PetClawSolver1D(kernelsType = 'P')
solver.dt = 0.0004
solver.max_steps = 5000
solver.set_riemann_solver('vc_advection')
solver.order = 2
solver.mthlim = 4
solver.dt_variable = False #Amal: need to handle the case dt_variable.


use_controller = True

if(use_controller):

# Controller instantiation
    claw = Controller()
    claw.outdir = './_output'
    claw.keep_copy = True
    claw.nout = 100
    claw.outstyle = 1
    claw.output_format = 'petsc'
    claw.tfinal = 1.0
    claw.solutions['n'] = init_solution
    claw.solver = solver

    # Solve
    status = claw.run()


    if claw.keep_copy:
    
        for n in xrange(0,11):
            sol = claw.frames[n]
            plotTitle="time: {0}".format(sol.t)
            viewer = PETSc.Viewer()
            viewer.createDraw(  title = plotTitle,  comm=sol.grid.gqVec.comm)


        
            OptDB = PETSc.Options()
            OptDB['draw_pause'] = -1
            sol.grid.gqVec.view(viewer)

else:
    sol = {"n":init_solution}
    
    solver.evolve_to_time(sol,.4)
    sol = sol["n"]

    viewer = PETSc.Viewer.DRAW(grid.gqVec.comm)
    OptDB = PETSc.Options()
    OptDB['draw_pause'] = -1
    viewer(grid.gqVec)
