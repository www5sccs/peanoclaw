#include "peanoclaw/repositories/RepositoryFactory.h"

#include "peanoclaw/repositories/RepositoryArrayStack.h"
#include "peanoclaw/repositories/RepositorySTDStack.h"

#include "peanoclaw/records/RepositoryState.h"

#ifdef Parallel
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/Partitioner.h"
#endif


peanoclaw::repositories::RepositoryFactory::RepositoryFactory() {
  #ifdef Parallel
  peano::parallel::Partitioner::initDatatypes();

  peanoclaw::State::initDatatype();
  peanoclaw::Vertex::initDatatype();
  peanoclaw::Cell::initDatatype();

  if (peanoclaw::records::RepositoryState::Datatype==0) {
    peanoclaw::records::RepositoryState::initDatatype();
  }
  #endif
}


peanoclaw::repositories::RepositoryFactory::~RepositoryFactory() {
}


void peanoclaw::repositories::RepositoryFactory::shutdownAllParallelDatatypes() {
  #ifdef Parallel
  peano::parallel::Partitioner::shutdownDatatypes();

  peanoclaw::State::shutdownDatatype();
  peanoclaw::Vertex::shutdownDatatype();
  peanoclaw::Cell::shutdownDatatype();

  if (peanoclaw::records::RepositoryState::Datatype!=0) {
    peanoclaw::records::RepositoryState::shutdownDatatype();
    peanoclaw::records::RepositoryState::Datatype = 0;
  }
  #endif
}


peanoclaw::repositories::RepositoryFactory& peanoclaw::repositories::RepositoryFactory::getInstance() {
  static peanoclaw::repositories::RepositoryFactory singleton;
  return singleton;
}

    
peanoclaw::repositories::Repository* 
peanoclaw::repositories::RepositoryFactory::createWithArrayStackImplementation(
  peano::geometry::Geometry&                   geometry,
  const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
  const tarch::la::Vector<DIMENSIONS,double>&  computationalDomainOffset,
  int                                          maxCellStackSize,    
  int                                          maxVertexStackSize,    
  int                                          maxTemporaryVertexStackSize    
) {
  #ifdef Parallel
  if (!tarch::parallel::Node::getInstance().isGlobalMaster()) {
    return new peanoclaw::repositories::RepositoryArrayStack(geometry, domainSize, computationalDomainOffset,maxCellStackSize,maxVertexStackSize,maxTemporaryVertexStackSize);
  }
  else
  #endif
  return new peanoclaw::repositories::RepositoryArrayStack(geometry, domainSize, computationalDomainOffset,maxCellStackSize,maxVertexStackSize,maxTemporaryVertexStackSize);
}    


peanoclaw::repositories::Repository* 
peanoclaw::repositories::RepositoryFactory::createWithSTDStackImplementation(
  peano::geometry::Geometry&                   geometry,
  const tarch::la::Vector<DIMENSIONS,double>&  domainSize,
  const tarch::la::Vector<DIMENSIONS,double>&  computationalDomainOffset
) {
  #ifdef Parallel
  if (!tarch::parallel::Node::getInstance().isGlobalMaster()) {
    return new peanoclaw::repositories::RepositorySTDStack(geometry);
  }
  else
  #endif
  return new peanoclaw::repositories::RepositorySTDStack(geometry, domainSize, computationalDomainOffset);
}
