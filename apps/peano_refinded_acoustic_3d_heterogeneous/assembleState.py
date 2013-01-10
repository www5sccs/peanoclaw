#!/usr/bin/env python
# encoding: utf-8

def assemblePeanoClawState(peanoClawSolution, globalDomain, num_eqn, mgrid):
  import numpy
  from clawpack.pyclaw.geometry import Dimension, Domain
  from clawpack.pyclaw.state import State

  dimensions = []
  for d in xrange(len(globalDomain.patch.dimensions)):
    dimensions.append(Dimension(globalDomain.patch.dimensions[d].name, globalDomain.patch.dimensions[d].lower, \
                                globalDomain.patch.dimensions[d].upper, mgrid))


  assembledPeanoClawState = State(Domain(dimensions), num_eqn)
  assembledPeanoClawState.q[0,:,:,:] = 0.0
  
  for peanoClawState in peanoClawSolution.gathered_states:
    offsetX = int((peanoClawState.patch.dimensions[0].lower - assembledPeanoClawState.patch.dimensions[0].lower) /
                  assembledPeanoClawState.patch.dimensions[0].delta )
    offsetY = int((peanoClawState.patch.dimensions[1].lower - assembledPeanoClawState.patch.dimensions[1].lower) /
                  assembledPeanoClawState.patch.dimensions[1].delta )
    offsetZ = int((peanoClawState.patch.dimensions[2].lower - assembledPeanoClawState.patch.dimensions[2].lower) /
                  assembledPeanoClawState.patch.dimensions[2].delta )
    assembledPeanoClawState.q[0, offsetX:offsetX+peanoClawState.q.shape[1], offsetY:offsetY+peanoClawState.q.shape[2],offsetZ:offsetZ+peanoClawState.q.shape[3]] = peanoClawState.q[0, :, :, :]
    
  # if numpy.min(assembledPeanoClawState.q[0, :, :,:]) < 0.0:
  #   raise Exception("Not all cells have been filled!")
    
  return assembledPeanoClawState.q[0,:,:,:]



