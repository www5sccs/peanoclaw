#!/usr/bin/env python
# encoding: utf-8
r"""
Module containing petclaw grid.

:Authors:
    Amal Alghamdi
    David Ketcheson
    Aron Ahmadia
"""
# ============================================================================
#      Copyright (C) 2010 David I. Ketcheson <david.ketcheson@kaust.edu.sa>
#
#  Distributed under the terms of the Berkeley Software Distribution (BSD) 
#  license
#                     http://www.opensource.org/licenses/
# ============================================================================

import copy

import numpy as np

from pyclaw.data import Data
from pyclaw.solution import Dimension, Grid
from petsc4py import PETSc

# ============================================================================
#  Default function definitions
# ============================================================================

# Default mapc2p function
def default_mapc2p(grid,x):
    r"""
    Returns the physical coordinate of the point x
    
    This is the stub function which simply returns the identity
    """
    return x
    



# ============================================================================
#  Dimension Object
# ============================================================================
class PCDimension(Dimension):
    r"""
    Basic class representing a dimension of a Grid object

    The only difference between PyClaw and PetClaw grids are the
    boundary conditions.
    
    :Initialization:
    
    Input:
     - *name* - (string) string Name of dimension
     - *lower* - (float) Lower extent of dimension
     - *upper* - (float) Upper extent of dimension
     - *n* - (int) Number of grid cells
     - *units* - (string) Type of units, used for informational purposes only
     - *mthbc_lower* - (int) Lower boundary condition method to be used
     - *mthbc_upper* - (int) Upper boundary condition method to be used
     
        
    Output:
     - (:class:`Dimension`) - Initialized Dimension object
    """

    def edge():
        doc = r"""(ndarrary(:)) - Location of all grid cell edge coordinates
        for this dimension"""
        def fget(self): 
            if self._edge is None:
                self._edge = np.empty(self.nend-self.nstart+1)
                for i in xrange(self.nstart,self.nend+1):
                    self.edge[i] = self.lower + i*self.d
            return self._edge
        return locals()
    edge = property(**edge())
    _edge = None

    def centerghost():
        doc = r"""(ndarrary(:)) - Location of all grid cell center coordinates
        for this dimension, including ghost cells"""
        def fget(self): 
            mbc=self.mbc
            if self._centerghost is None:
                self._centerghost = np.empty(self.nend-self.nstart+2*mbc)
                for i in xrange(self.nstart-mbc,self.nend+mbc):
                    self.centerghost[i-self.nstart+mbc] = self.lower + (i+0.5)*self.d
            return self._centerghost
        return locals()
    centerghost = property(**centerghost())
    _centerghost = None

    def center():
        doc = r"""(ndarrary(:)) - Location of all grid cell center coordinates
        for this dimension"""
        def fget(self): 
            if self._center is None:
                self._center = np.empty(self.nend-self.nstart)
                for i in xrange(self.nstart,self.nend):
                    self.center[i-self.nstart] = self.lower + (i+0.5)*self.d
            return self._center
        return locals()
    center = property(**center())
    _center = None

    # ========== Setting Boundary Conditions ==================================
    def qbc_lower(self,grid,qbc):
        r"""
        
        """
        # User defined functions
        if self.mthbc_lower == 0:
            self.user_bc_lower(grid,self,qbc)
        # Zero-order extrapolation
        elif self.mthbc_lower == 1:
            ##specify rank
            rank = PETSc.Comm.getRank(PETSc.COMM_WORLD) # Amal: hardcoded communicator
            if rank == 0:
                qbc[:grid.mbc,...] = qbc[grid.mbc,...]
        # Periodic
        elif self.mthbc_lower == 2:
            pass # Amal: this is implemented automatically by petsc4py
            
        # Solid wall bc
        elif self.mthbc_lower == 3:
            raise NotImplementedError("Solid wall upper boundary condition not implemented.")
        else:
            raise NotImplementedError("Boundary condition %s not implemented" % x.mthbc_lower)


    # ========== Setting Boundary Conditions ==================================
    def qbc_upper(self,grid,qbc):
        r"""
        
        """
        # User defined functions
        if self.mthbc_upper == 0:
            self.user_bc_upper(grid,self,qbc)
        # Zero-order extrapolation
        elif self.mthbc_upper == 1:
            rank = PETSc.Comm.getRank(PETSc.COMM_WORLD) # Amal: hardcoded communicator
            size = PETSc.Comm.getSize(PETSc.COMM_WORLD)
            
            if rank == size-1:
                local_n = grid.q.shape[0]
                list_from =[local_n - grid.mbc -1]*grid.mbc    
                list_to = range(local_n - grid.mbc, local_n )
                grid.q[list_to,:]=grid.q[list_from,:]
 	    
        elif self.mthbc_upper == 2:
            # Periodic
            pass # Amal: this is implemented automatically by petsc4py

        # Solid wall bc
        elif self.mthbc_upper == 3:
            raise NotImplementedError("Solid wall upper boundary condition not implemented.")

        else:
            raise NotImplementedError("Boundary condition %s not implemented" % x.mthbc_lower)



# ============================================================================
#  petclaw Grid object definition
# ============================================================================
class PCGrid(Grid):
    r"""
    Basic representation of a single grid in petclaw

    The only difference between PetClaw grid and PyClaw grid is
    the definition of q(), local_n(), qbc() __getstate__(), 
    and __setstate__().
    
    :Dimension information:
    
        Each dimension has an associated name with it that can be accessed via
        that name such as ``grid.x.n`` which would access the x dimension's
        number of grid cells.
    
    :Global Grid information:
    
        Each grid has a value for :attr:`level`, :attr:`gridno`, :attr:`t`, 
        :attr:`mbc`, :attr:`meqn` and :attr:`aux_global`.  These correspond
        to global grid traits and determine many of the properties and sizes
        of the data arrays.
        
    :Grid Data:
    
        The arrays :attr:`q`, :attr:`aux` and :attr:`capa` have variable 
        extents based on the set of dimensions present and the values of 
        :attr:`meqn` and :attr:`maux`.  Note that these are initialy set to 
        None so need to be instantiated.  For convenience, the methods
        :meth:`emtpy_q`, :meth:`ones_q`, and :meth:`zeros_q` for ``q`` and
        :meth:`emtpy_aux`, :meth:`ones_aux`, and :meth:`zeros_aux` for ``aux``
        are provided to initialize these arrays.  The :attr:`capa` array is 
        initially set to all ``1.0`` and needs to be manually set.
        
    :Properties:

        If the requested property has multiple values, a list will be returned
        with the corresponding property belonging to the dimensions in order.
         
    :Initialization:
    
        Input:
         - *dimensions* - (list of :class:`Dimension`) Dimensions that are to 
           be associated with this grid
            
        Output:
         - (:class:`Grid`) Initialized grid object
    """

    # Serialization Definitions (save everything but q and aux!)

    def __getstate__(self):
        doc = r"""Returns dictionary of serializable attributes of this object"""
        #only need a shallow copy here
        result = self.__dict__.copy()
        del result['gqVec']
        del result['gauxVec']
        del result['q_da']
        del result['aux_da']
        del result['lqVec']
        del result['lauxVec']
        
        return result

    def __setstate__(self, state):
        doc = r"""Reconstructs this object from a dictionary of its serializable attributes"""
        self.__dict__ = state

        # these are all in a bad state and need to be explicitly loaded from viewers
        self.q_da = None
        self.aux_da = None
        self.gqVec = None
        self.lqVec = None
        self.gauxVec = None
        self.lauxVec = None
        
    
    # ========== Property Definitions ========================================
    def local_n():
        def fget(self):
            #Amal doc
            shape = []
            ranges = self.da.getRanges()
 
            for i in ranges:
                shape.append(i[1]-i[0])
            return shape
        return locals()
    def q():
        def fget(self):
            #THIS ONLY WORKS IN 1D:
            q=self.gqVec.getArray().reshape([-1,self.meqn])
            return q
        def fset(self,q):
            if self.gqVec is None:
                self.init_q_petsc_structures()
            self.gqVec.setArray(q.reshape([-1]))
            self.q_da.globalToLocal(self.gqVec, self.lqVec)
        return locals()

    local_n = property(**local_n())
    q = property(**q())
    
    # ========== Class Methods ===============================================
    def __init__(self,dimensions):
        r"""
        Instantiate a PCGrid object

        Here we duplicate the __init__ function from the parent class Grid.
        
        Really we should just do this:

        super(PCGrid,self).__init__(dimensions)

        But the problem is that Grid.__init__() sets q=None, messing up
        our use of q as a property.  We should find a better way to
        resolve this.

        See :class:`PCGrid` for more info.
        """
        
        # ========== Attribute Definitions ===================================
        self.level = 1
        r"""(int) - AMR level this grid belongs to, ``default = 1``"""
        self.gridno = 1
        r"""(int) - Grid number of current grid, ``default = 0``"""
        self.t = 0.0
        r"""(float) - Current time represented on this grid, 
            ``default = 0.0``"""
        self.mbc = 2
        r"""(int) - Number of ghost cells along the boundaries, 
            ``default = 2``"""
        self.meqn = 1
        r"""(int) - Dimension of q array for this grid, ``default = 1``"""
        self.aux = None
        r"""(ndarray(...,maux)) - Auxiliary array for this grid containing per 
            cell information"""
        self.capa = None
        r"""(ndarray(...)) - Capacity array for this grid, ``default = 1.0``"""
        self.aux_global = {}
        r"""(dict) - Dictionary of global values for this grid, 
            ``default = {}``"""
        self.mapc2p = default_mapc2p
        r"""(func) - Grid mapping function"""

        ###  Some PETSc4Py specific stuff
        self.q_da = None
        self.gqVec = None
        self.lqVec = None

        # Dimension parsing
        if isinstance(dimensions,Dimension):
            dimensions = [dimensions]
        self._dimensions = []
        for dim in dimensions:
            self.add_dimension(dim)


    def init_q_petsc_structures(self):
        r"""
        Initilizes PETSc structures for q. It initilizes q_da, gqVec and lqVec
        
        """

        periodic = False
        for dimension in self.dimensions:
            if dimension.mthbc_lower == 2 or dimension.mthbc_upper == 2:
                periodic = True
                break
                
        if self.ndim == 1:
            if periodic: periodic_type = PETSc.DA.PeriodicType.X
            else: periodic_type = PETSc.DA.PeriodicType.GHOSTED_XYZ
        elif self.ndim == 2:
            if periodic: periodic_type = PETSc.DA.PeriodicType.XY
            else: periodic_type = PETSc.DA.PeriodicType.GHOSTED_XYZ
        elif self.ndim == 3:
            if periodic: periodic_type = PETSc.DA.PeriodicType.XYZ
            else: periodic_type = PETSc.DA.PeriodicType.GHOSTED_XYZ
        else:
            raise Exception("Invalid number of dimensions")

        self.q_da = PETSc.DA().create(dim=self.ndim,
                                    dof=self.meqn,
                                    sizes=self.n, 
                                    periodic_type = periodic_type,
                                    #stencil_type=self.STENCIL,
                                    stencil_width=self.mbc,
                                    comm=PETSc.COMM_WORLD)
        self.gqVec = self.q_da.createGlobalVector()
        self.lqVec = self.q_da.createLocalVector()

        #Now set up the local indices:
        ranges = self.q_da.getRanges()
        for i,range in enumerate(ranges):
            self.dimensions[i].nstart=range[0]
            self.dimensions[i].nend  =range[1]
            
    
    # ========== Grid Operations =============================================
    # Convenience routines for initialization of q and aux
    
    def qbc(self):
        #Apply BCs here
        #THIS ONLY WORKS IN 1D:
        qbc=self.lqVec.getArray().reshape([-1,self.meqn])
        #return qbc <--- BUG!
        for i in xrange(len(self._dimensions)):
            dim = getattr(self,self._dimensions[i])
            #If a user defined boundary condition is being used, send it on,
            #otherwise roll the axis to front position and operate on it
            if dim.mthbc_lower == 0:
                dim.qbc_lower(self,qbc)
            else:
                dim.qbc_lower(self,np.rollaxis(qbc,i))
            if dim.mthbc_upper == 0:
                dim.qbc_upper(self,qbc)
            else:
                dim.qbc_upper(self,np.rollaxis(qbc,i))
        return qbc
