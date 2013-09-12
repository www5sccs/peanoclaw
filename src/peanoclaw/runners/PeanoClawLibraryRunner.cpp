/*
 * PeanoClawLibraryRunner.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/runners/PeanoClawLibraryRunner.h"

#include "peanoclaw/runners/Runner.h"
#include "peanoclaw/repositories/RepositoryFactory.h"
#include "peanoclaw/State.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/VertexDescription.h"
#include "peanoclaw/records/Data.h"
#include "peanoclaw/records/Cell.h"
#include "peanoclaw/records/Vertex.h"
#include "peano/heap/Heap.h"

#include "peano/utils/UserInterface.h"
#include "peano/peano.h"

#if defined(Parallel)
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/SendReceiveBufferPool.h"
#include "peano/parallel/JoinDataBufferPool.h"
#include "peano/parallel/messages/ForkMessage.h"
#include "tarch/parallel/FCFSNodePoolStrategy.h"
#include "peano/parallel/loadbalancing/Oracle.h"
#include "peano/parallel/loadbalancing/OracleForOnePhaseWithGreedyPartitioning.h"
#endif

#ifdef SharedTBB
#include "tarch/multicore/tbb/Core.h"
#endif

#include "peano/datatraversal/autotuning/Oracle.h"
#include "peano/datatraversal/autotuning/OracleForOnePhaseDummy.h"
#include "queries/QueryServer.h"
#include "queries/records/HeapQuery.h"
tarch::logging::Log peanoclaw::runners::PeanoClawLibraryRunner::_log("peanoclaw::runners::PeanoClawLibraryRunner");

peanoclaw::runners::PeanoClawLibraryRunner::PeanoClawLibraryRunner(
  peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid& configuration,
  peanoclaw::Numerics& numerics,
  const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
  const tarch::la::Vector<DIMENSIONS, double>& domainSize,
  const tarch::la::Vector<DIMENSIONS, double>& initialMinimalMeshWidth,
  const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
  int defaultGhostLayerWidth,
  int unknownsPerSubcell,
  int auxiliarFieldsPerSubcell,
  double initialTimestepSize,
  bool useDimensionalSplittingOptimization
) :
  _plotNumber(1),
  _configuration(configuration),
  _iterationTimer("peanoclaw::runners::PeanoClawLibraryRunner", "iteration", false),
  _totalRuntime(0.0),
  _numerics(numerics),
  _validateGrid(false)
{
  #ifndef Asserts
  _validateGrid = false;
  #endif

  //Parallel configuration
  #ifdef Parallel
  tarch::parallel::Node::getInstance().setTimeOutWarning(30);
  tarch::parallel::Node::getInstance().setDeadlockTimeOut(60);
  #endif

  //Multicore configuration
  #ifdef SharedMemoryParallelisation
  tarch::multicore::tbb::Core::getInstance().configure(1);
  #endif

  //User interface
  peano::utils::UserInterface userInterface;
  userInterface.writeHeader();

  //Initialize heap data
  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().setName("CellDescription");
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().setName("Data");
  peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance().setName("VertexDescription");

  initializeParallelEnvironment();

  peano::datatraversal::autotuning::Oracle::getInstance().setOracle( new peano::datatraversal::autotuning::OracleForOnePhaseDummy(true) );

  //Initialize pseudo geometry
  _geometry =
    new  peano::geometry::Hexahedron (
      domainSize,  // width
      domainOffset // offset
    );
  _repository =
    peanoclaw::repositories::RepositoryFactory::getInstance().createWithSTDStackImplementation(
      *_geometry,
      domainSize,   // domainSize,
      domainOffset  // computationalDomainOffset
    );

  logInfo("PeanoClawLibraryRunner", "Initial values: "
      << "Domain size = [" << domainSize << "]"
      << ", default subdivision factor = " << subdivisionFactor
      << ", default ghostlayer width = " << defaultGhostLayerWidth
      << ", unknowns per cell = " << unknownsPerSubcell);

  assertion(_repository != 0);
  State& state = _repository->getState();
  state.setDefaultSubdivisionFactor(subdivisionFactor);
  state.setDefaultGhostLayerWidth(defaultGhostLayerWidth);
  state.setUnknownsPerSubcell(unknownsPerSubcell);
  state.setAuxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell);
  tarch::la::Vector<DIMENSIONS, double> initialMinimalSubcellSize = tarch::la::multiplyComponents(initialMinimalMeshWidth, subdivisionFactor.convertScalar<double>());
  state.setInitialMinimalMeshWidth(initialMinimalSubcellSize);
  state.setNumerics(numerics);
  state.resetTotalNumberOfCellUpdates();
  state.setInitialTimestepSize(initialTimestepSize);
  state.setDomain(domainOffset, domainSize);
  state.setUseDimensionalSplittingOptimization(useDimensionalSplittingOptimization && !_configuration.disableDimensionalSplittingOptimization());

  //Initialise Grid (two iterations needed to set the initial ghostlayers of patches neighboring refined patches)
  state.setIsInitializing(true);

#ifdef Parallel
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
     tarch::parallel::NodePool::getInstance().waitForAllNodesToBecomeIdle();  
#endif


  if(_validateGrid) {
    _repository->switchToInitialiseAndValidateGrid();
  } else {
    _repository->switchToInitialiseGrid();
  }
  _repository->iterate(); _repository->iterate(); _repository->iterate(); _repository->iterate();
  do {
    state.setInitialRefinementTriggered(false);
    if(_validateGrid) {
      _repository->switchToInitialiseAndValidateGrid();
    } else {
      _repository->switchToInitialiseGrid();
    }
    _repository->iterate();
  } while(state.getInitialRefinementTriggered());
  state.setIsInitializing(false);

  _repository->getState().setPlotNumber(0);
  if(_configuration.plotAtOutputTimes() || _configuration.plotSubsteps()) {
    _repository->switchToPlot(); _repository->iterate();
  }

#ifdef Parallel
  }
#endif
  if(tarch::parallel::Node::getInstance().isGlobalMaster()){
     char* hostname = getenv("PEANOCLAW_HOSTNAME");
      
     _queryServer=new de::tum::QueryCxx2SocketPlainPort(hostname,50000,8192);

  
  }
}

peanoclaw::runners::PeanoClawLibraryRunner::~PeanoClawLibraryRunner()
{
  logTraceIn("~PeanoClawLibraryRunner");
//  logInfo("~PeanoClawLibraryRunner()", "Total number of cell updates: " << _repository->getState().getTotalNumberOfCellUpdates());

  //Delete remaining heap data
  peano::heap::Heap<peanoclaw::records::CellDescription>& cellDescriptionHeap = peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance();
//  if(cellDescriptionHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for CellDescriptions still contains " << cellDescriptionHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  peano::heap::Heap<peanoclaw::records::Data>& dataHeap = peano::heap::Heap<peanoclaw::records::Data>::getInstance();
//  if(dataHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for patch data still contains " << dataHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  peano::heap::Heap<peanoclaw::records::VertexDescription>& vertexDescriptionHeap = peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance();
//  if(vertexDescriptionHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for VertexDescriptions still contains " << vertexDescriptionHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  cellDescriptionHeap.deleteAllData();
  dataHeap.deleteAllData();
  vertexDescriptionHeap.deleteAllData();

  #ifdef Parallel
  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
  #endif
    _repository->getState().plotTotalStatistics();

    _repository->logIterationStatistics();
    _repository->terminate();
  #ifdef Parallel
  }
  #endif
  delete _repository;
  delete _geometry;
  delete _queryServer;
  #ifdef Parallel
  tarch::parallel::NodePool::getInstance().terminate();
  #endif
  peano::shutdownParallelEnvironment();
  peano::shutdownSharedMemoryEnvironment();

  logTraceOut("~PeanoClawLibraryRunner");
}

 
void peanoclaw::runners::PeanoClawLibraryRunner::sync(){
	int parts=-1;
	std::cout<<"syncing with query server"<<std::endl;
	_queryServer->getNumberOfParts(parts);
        if(parts>0){	
		
		int *mids=new int[parts];
		queries::records::HeapQuery q;
		double offset[2];
		double size[2];
		int dim[2];		
		_queryServer->getQueryDescription(offset,2,size,2,dim,2,mids,1);
		tarch::la::Vector<3,double> la_offset;
		tarch::la::Vector<3,double> la_size;
		tarch::la::Vector<3,int> la_dim;
				
		la_offset[0]=offset[0];
		la_offset[1]=offset[1];
		la_size[0] = size[0];
		la_size[1] = size[1];
		la_dim[0] = dim[0];
		la_dim[1] = dim[1];
		q.setOffset(la_offset);
		q.setSize(la_size);		
		q.setDimenions(la_dim);
		std::cout<<"received q:"<<q.getOffset()<<","<<q.getSize()<<std::endl;
				
		
		queries::QueryServer::getInstance().addQuery(q);
		delete []mids;
	}
}
void peanoclaw::runners::PeanoClawLibraryRunner::evolveToTime(
  double time
) {
  logTraceIn("evolveToTime");
  
  bool plotSubsteps = _configuration.plotSubsteps()
      || (_configuration.plotSubstepsAfterOutputTime() != -1 && _configuration.plotSubstepsAfterOutputTime() <= _plotNumber);
 
  _repository->getState().setNumerics(_numerics);

  _repository->getState().setGlobalTimestepEndTime(time);
  _repository->getState().setNumerics(_numerics);
  _repository->getState().setPlotNumber(_plotNumber);
  int c=0;
  do {
    logInfo("evolveToTime", "Solving timestep " << (_plotNumber-1) << " with maximum global time interval ("
        << _repository->getState().getStartMaximumGlobalTimeInterval() << ", " << _repository->getState().getEndMaximumGlobalTimeInterval() << ")"
        << " and minimum global time interval (" << _repository->getState().getStartMinimumGlobalTimeInterval() << ", " << _repository->getState().getEndMinimumGlobalTimeInterval() << ")");
    _iterationTimer.startTimer();

    _repository->getState().resetGlobalTimeIntervals();
    _repository->getState().resetMinimalTimestep();
    _repository->getState().setAllPatchesEvolvedToGlobalTimestep(true);

    if(plotSubsteps) {
      
      _repository->getState().setPlotNumber(_plotNumber);
      if(_validateGrid) {
        _repository->switchToSolveTimestepAndPlotAndValidateGrid();
      } else {
        _repository->switchToSolveTimestepAndPlot();
      }
      _repository->iterate();
      _plotNumber++;
      
    } else {
      if(_validateGrid) {
        _repository->switchToSolveTimestepAndValidateGrid();
      } else {
        _repository->switchToSolveTimestep();
      }
      _repository->iterate();
      if(tarch::parallel::NodePool::getInstance().areAllNodesWorking()&&c==0){      
      	sync();
      	c++;
      }
      _repository->switchToQuery();
      _repository->iterate();
      	
    }

    _repository->getState().plotStatisticsForLastGridIteration();

    _iterationTimer.stopTimer();
    _totalRuntime += (double)_iterationTimer.getCPUTicks() / (double)CLOCKS_PER_SEC;
    logInfo("evolveToTime", "Wallclock time for this grid iteration/Total runtime: " << _iterationTimer.getCalendarTime() << "s/" << _totalRuntime << "s");
    logInfo("evolveToTime", "Minimal timestep for this grid iteration: " << _repository->getState().getMinimalTimestep());

    assertion(_repository->getState().getMinimalTimestep() < std::numeric_limits<double>::infinity());
  } while(!_repository->getState().getAllPatchesEvolvedToGlobalTimestep());

  if(_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _repository->getState().setPlotNumber(_plotNumber);
    _repository->switchToPlot(); _repository->iterate();
    _plotNumber++;
  } else if (!_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _plotNumber++;
  }
  logTraceOut("evolveToTime");
}

void peanoclaw::runners::PeanoClawLibraryRunner::initializeParallelEnvironment() {
  #if defined(Parallel)
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    tarch::parallel::NodePool::getInstance().setStrategy( new tarch::parallel::FCFSNodePoolStrategy() );
  }
  tarch::parallel::NodePool::getInstance().restart();

  peano::parallel::loadbalancing::Oracle::getInstance().setOracle(
    new peano::parallel::loadbalancing::OracleForOnePhaseWithGreedyPartitioning(true)
  );

  // have to be the same for all ranks
  peano::parallel::SendReceiveBufferPool::getInstance().setBufferSize(64);
  peano::parallel::JoinDataBufferPool::getInstance().setBufferSize(64);
  #endif
}

void peanoclaw::runners::PeanoClawLibraryRunner::gatherCurrentSolution() {
  logTraceIn("gatherCurrentSolution");
  assertion(_repository != 0);

  _repository->switchToGatherCurrentSolution();
  _repository->iterate();
  logTraceOut("gatherCurrentSolution");
}

int peanoclaw::runners::PeanoClawLibraryRunner::runWorker() {
  #ifdef Parallel
  int newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob();
  while ( newMasterNode != tarch::parallel::NodePool::JobRequestMessageAnswerValues::Terminate ) {
    if ( newMasterNode >= tarch::parallel::NodePool::JobRequestMessageAnswerValues::NewMaster ) {
      peano::parallel::messages::ForkMessage forkMessage;
      forkMessage.receive(tarch::parallel::NodePool::getInstance().getMasterRank(),tarch::parallel::NodePool::getInstance().getTagForForkMessages(), true);
      _repository->restart(
        forkMessage.getH(),
        forkMessage.getDomainOffset(),
        forkMessage.getLevel(),
        forkMessage.getPositionOfFineGridCellRelativeToCoarseGridCell()
      );

      bool continueToIterate = true;
      while (continueToIterate) {
        switch (_repository->continueToIterate()) {
          case peanoclaw::repositories::Repository::Continue:
            _repository->iterate();
            break;
          case peanoclaw::repositories::Repository::Terminate:
            continueToIterate = false;
            break;
          case peanoclaw::repositories::Repository::RunGlobalStep:
            //runGlobalStep();
            break;
        }
      }

      // insert your postprocessing here
      // -------------------------------

      // -------------------------------

      _repository->terminate();
    }
    else if ( newMasterNode == tarch::parallel::NodePool::JobRequestMessageAnswerValues::RunAllNodes ) {
      //runGlobalStep();
    }
    newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob();
  }

  #endif
  return 0;
}
