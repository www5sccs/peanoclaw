// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef PEANOCLAW_MAPPINGS_Remesh_H_
#define PEANOCLAW_MAPPINGS_Remesh_H_

#include "peanoclaw/Cell.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/State.h"
#include "peanoclaw/Vertex.h"
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.h"
#include "peanoclaw/parallel/NeighbourCommunicator.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/VertexDescription.h"
#include "peanoclaw/records/Data.h"
#include "peanoclaw/statistics/ParallelStatistics.h"
#include "peanoclaw/statistics/LevelStatistics.h"

#include "peano/grid/VertexEnumerator.h"
#include "peano/MappingSpecification.h"

#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"

#include "tarch/multicore/MulticoreDefinitions.h"

namespace peanoclaw {

  namespace interSubgridCommunication {
    class GridLevelTransfer;
  }

  namespace mappings {
    class Remesh;
  }

  class Numerics;
  class Patch;
}


/**
 * This is a mapping from the spacetree traversal events to your user-defined activities.
 * The latter are realised within the mappings. 
 * 
 * @author Peano Development Toolkit (PDT) by  Tobias Weinzierl
 * @version $Revision: 1.10 $
 */
class peanoclaw::mappings::Remesh {
  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::VertexDescription VertexDescription;
    typedef peanoclaw::records::Data Data;
//    typedef peanoclaw::statistics::LevelStatistics LevelStatistics;

    /**
     * Map from a hanging node's position and level to
     */
    static peanoclaw::interSubgridCommunication::aspects::AdjacentSubgrids::VertexMap _vertexPositionToIndexMap;
    static peanoclaw::parallel::NeighbourCommunicator::RemoteSubgridMap               _remoteSubgridMap;

    int _unknownsPerSubcell;

    int _auxiliarFieldsPerSubcell;

    tarch::la::Vector<DIMENSIONS, int> _defaultSubdivisionFactor;

    int _defaultGhostLayerWidth;

    double _initialTimestepSize;

    peanoclaw::Numerics* _numerics;

    tarch::la::Vector<DIMENSIONS, double> _domainOffset;

    tarch::la::Vector<DIMENSIONS, double> _domainSize;

    peanoclaw::interSubgridCommunication::GridLevelTransfer* _gridLevelTransfer;

    tarch::la::Vector<DIMENSIONS, double> _initialMinimalMeshWidth;

    bool _isInitializing;

    bool _useDimensionalSplittingOptimization;

    peanoclaw::statistics::ParallelStatistics _parallelStatistics;
    peanoclaw::statistics::ParallelStatistics _totalParallelStatistics;

    peanoclaw::State const* _state;

    int _iterationNumber;

    //Watches
    static tarch::timing::Watch _spacetreeCommunicationWaitingTimeWatch;

  public:
    /**
     * These flags are used to inform Peano about your operation. It tells the 
     * framework whether the operation is empty, whether it works only on the 
     * spacetree leaves, whether the operation can restart if the thread 
     * crashes (resiliency), and so forth. This information allows Peano to
     * optimise the code.
     */
    static peano::MappingSpecification   touchVertexLastTimeSpecification();
    static peano::MappingSpecification   touchVertexFirstTimeSpecification();
    static peano::MappingSpecification   enterCellSpecification();
    static peano::MappingSpecification   leaveCellSpecification();
    static peano::MappingSpecification   ascendSpecification();
    static peano::MappingSpecification   descendSpecification();


    /**
     * Mapping constructor.
     *
     * Mappings have to have a standard constructor and, typically, no other 
     * constructor does exist. While the constructor may initialise a mapping, 
     * Peano's concept requires the mapping to be semi-stateless:
     *
     * - At construction time the mapping has no well-defined state, i.e. the 
     *   values set by the constructor are meaningless.
     * - Whenever the mapping's beginIteration() operation is called, the 
     *   mapping has to initialise itself. To do this, it has to analyse the 
     *   passed state object. The beginIteration() operation may set attributes 
     *   of the mapping and these attributes now have a valid state.
     * - All the subsequent calls on the mapping can rely on valid mapping 
     *   attributes until
     * - The operation endIteration() is invoked. Afterwards, all the mapping's 
     *   attributes have an undefined state.
     *
     * With this concept, you cannot ensure a consistent mapping state 
     * in-between two iterations: While the first iteration might set some 
     * mapping attributes, the attributes become invalid after the first 
     * endIteration() call and might be changed from outside before the next 
     * beginIteration() is invoked.
     *
     * To implement persistent attributes, you have to write back all these  
     * attributes at endIteration() and reload them at the next beginIteration() 
     * call. With this sometimes confusing persistency concept, we can ensure 
     * that your code works on a parallel machine and for any mapping/algorithm 
     * modification.
     */
    Remesh();

    #if defined(SharedMemoryParallelisation)
    /**
     * Copy constructor for multithreaded code
     *
     * If Peano uses several cores, the mappings are duplicated due to this 
     * copy constructor.
     *
     * This operation is thread-safe, i.e. you need no semaphores here. 
     *
     * @see mergeWithWorkerThread()
     */
    Remesh(const Remesh& masterThread);
    #endif

    /**
     * Destructor. Typically does not implement any operation.
     */
    virtual ~Remesh();
  
    #if defined(SharedMemoryParallelisation)
    /**
     * Merge with worker thread
     *
     * In Peano's multithreaded mode, each mapping is duplicated among the 
     * threads. This duplication is done via the copy constructor. At the end 
     * of a parallel section, the individual copies of the mappings are 
     * merged together into one the global state instance again. 
     * 
     * If there are t threads, Peano takes the original mapping and makes t 
     * copies. Then, the t copies are ran in parallel. In the end, the t copies 
     * are merged into the original step by step. If your mapping holds 
     * variables, also these variables are copied per thread. If you have 
     * pointers, be sure that those are copied in the constructor as well or 
     * handled correctly. This place then is the place to merge all the copies 
     * back into one mapping instance.
     *
     * This operation is thread-safe, i.e. you need no semaphores here.
     *
     * While this is the right place to merge parallel data, and while you 
     * do not need any semaphores at all here, there are also cases where this 
     * implementation pattern doesn't work. The best example is a plotter: 
     * You may not copy a plotter, and it does not make sense to merge a plotter. 
     * So, in this case, I recommend to hold a pointer to the plotter in the 
     * original mapping and to create the plotter on the heap. This pointer then 
     * is copied to all thread-local mappings and all threads use the same object
     * on the heap. However, you should protect this object by a BooleanSemaphore 
     * and a lock to serialise all accesses to the plotter.    
     */   
    void mergeWithWorkerThread(const Remesh& workerThread);
    #endif

    /**
     * Create an inner vertex.
     *
     * You receive a vertex, its position, and the size of its @f$ 2^d @f$ 
     * surrounding cells. The grid already has found out that this will be an 
     * inner vertex, however, you have to set valid vertices.
     *
     * The vertices have an attribute position or x, too. However, this 
     * attribute is available in Debug mode only and it should be used solely 
     * to implement assertions. To work with the vertex's position, use the 
     * attribute x instead.
     *
     * @image html peano/grid/geometry-vertex-inside-outside.png
     *
     * If you are working with moving geometries, an inner vertex is either 
     * created at startup time or it is a former boundary vertex becoming an 
     * inner vertex now. To recognise the latter case is up to you, as every
     * PDE reacts differently.
     *
     * It might seem to make sense to add an assertion such as 
     * \code 
 assertion1( isInside() || isBoundary(), *this );
 \endcode
     * at the very beginning of your implementation. However, such an assertion 
     * fails as Peano switches the vertex type after the initialisation routine. 
     * The reason for this behaviour is simple: For moving boundaries, often 
     * you'd like to know what type the vertex had before (was it outside or 
     * was it an inner point becoming a boundary/boundary point becoming an 
     * inner point). This is possible, if the type is switched after the 
     * initialisation.
     *
     * The refinement process cuts a coarse grid cell into pieces. You have 
     * access to this coarse grid data via coarseGridCell (coarse grid's cell
     * data), coarseGridVertices, and the corresponding 
     * coarseGridVerticesEnumerator. The coarse grid cell always is cut into 
     * @f$ 3^d @f$ subcubes, i.e. two cuts along each coordinate axis. The 
     * integer vector fineGridPositionOfVertex (each entry is in-between zero
     * and 4) tells you where within this local fine grid the new vertex is 
     * located. You may modify both the fine grid vertex and the coarse grid
     * vertices and cells. However, only the initialisation of fineGridVertex
     * is mandatory.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe.
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     *
     * @param fineGridVertex  Vertex that is to be initialised. This is the 
     *                        central object you should set values to.
     * @param fineGridX       Position of vertex.
     * @param fineGridH       Size of the surrounding cells of fineGridVertex.
     * @param coarseGridVertices Vertices of the coarser grid. These vertices 
     *                        have to be accessed using the enumerator.
     * @param coarseGridVerticesEnumerator Enumerator for coarseGridVertices. 
     *                        It also holds the coarse grid's cell size.
     * @param coarseGridCell  This it the cell of the coarser grid into which 
     *                        the new vertex is embedded.
     * @param fineGridPositionOfVertex  Position of new vertex within the 
     *                        fine grid.
     */
    void createInnerVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
    );


    /**
     * Create a boundary vertex.
     *
     * You receive a vertex, its position, and the size of its @f$ 2^d @f$ 
     * surrounding cells. The grid already has found out that this will be a 
     * boundary vertex, however, you have to set valid vertices. Boundary 
     * implies that this vertex has less than @f$ 2^d @f$ cells that lay 
     * completely with in the computational domain.
     *
     * @image html peano/grid/geometry-vertex-inside-outside.png
     *
     * The vertices have an attribute position or x, too. However, this 
     * attribute is available in Debug mode only and it should be used solely 
     * to implement assertions. To work with the vertex's position, use the 
     * attribute x instead. 
     *
     * A boundary vertex is 
     * - either created at startup time,
     * - a former outer vertex, or
     * - a former inner vertex.
     *      
     * The latter two cases happen if and only if you are working with moving 
     * geometries. To recognise the latter case is up to you, as every
     * PDE reacts differently, i.e. there's no boundary flag for the vertices 
     * a priori.
     *
     * It might seem to make sense to add an assertion such as 
     * \code 
 assertion1( isInside() || isBoundary(), *this );
 \endcode
     * at the very beginning of your implementation. However, such an assertion 
     * fails as Peano switches the vertex type after the initialisation routine. 
     * The reason for this behaviour is simple: For moving boundaries, often 
     * you'd like to know what type the vertex had before (was it outside or 
     * was it an inner point becoming a boundary/boundary point becoming an 
     * inner point). This is possible, if the type is switched after the 
     * initialisation.     
     *
     * The refinement process cuts a coarse grid cell into pieces. You have 
     * access to this coarse grid data via coarseGridCell (coarse grid's cell
     * data), coarseGridVertices, and the corresponding 
     * coarseGridVerticesEnumerator. The coarse grid cell always is cut into 
     * @f$ 3^d @f$ subcubes, i.e. two cuts along each coordinate axis. The 
     * integer vector fineGridPositionOfVertex (each entry is in-between zero
     * and 4) tells you where within this local fine grid the new vertex is 
     * located. You may modify both the fine grid vertex and the coarse grid
     * vertices and cells. However, only the initialisation of fineGridVertex
     * is mandatory.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe.
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     * 
     * @param fineGridVertex  Vertex that is to be initialised. This is the 
     *                        central object you should set values to.
     * @param fineGridX       Position of vertex.
     * @param fineGridH       Size of the surrounding cells of fineGridVertex.
     * @param coarseGridVertices Vertices of the coarser grid. These vertices 
     *                        have to be accessed using the enumerator.
     * @param coarseGridVerticesEnumerator Enumerator for coarseGridVertices. 
     *                        It also holds the coarse grid's cell size.
     * @param coarseGridCell  This it the cell of the coarser grid into which 
     *                        the new vertex is embedded.
     * @param fineGridPositionOfVertex  Position of new vertex within the 
     *                        fine grid.
     */
    void createBoundaryVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
    );


    /**
     * Create Hanging Vertex
     *
     * Peano treats hanging vertices as not persistent, i.e. they are not kept
     * in-between two iterations. Even more, it might happen that a hanging 
     * vertex is created and destroyed several times throughout an iteration. 
     * Each hanging vertex create call is accompanied by a destroy call. If a 
     * cell has hanging vertices, you can be sure that all adjacent vertices 
     * have been initialised before, i.e. for each adjacent vertex either 
     * touchVertexFirstTime() or createHangingVertex() has been called. 
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe.
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     */
    void createHangingVertex(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
    );


    /**
     * Counterpart of create operation
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe.
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     */
    void destroyHangingVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
    );


    /**
     * Destroy a vertex.
     *
     * For the regular grid, a vertex typically is destroyed due to moving
     * boundaries. The operation does not distinguish between boundary and
     * inner vertices, i.e. if you wanna treat them differently, you have to
     * implement this manually. The destory operation also is called after a 
     * vertex has been moved to a different node due to dynamic load balancing.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe.
     * 
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     *
     * @see createBoundaryVertex() for a argument description.
     */
    void destroyVertex(
      const peanoclaw::Vertex&   fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
    );


    /**
     * Create an inner cell.
     *
     * Whenever the grid management has created a new cell (either at startup or
     * due to moving boundaries), it afterwards invokes this operation. Here you 
     * can add your PDE-specific initialisation. The grid management already has 
     * found out that this cell will be an inner cell, so you don't have to 
     * doublecheck this again. Instead, you can focuse on PDE-specific stuff.
     *
     * @image html peano/grid/geometry-cell-inside-outside.png
     *
     * The vertices surrounding the cell already are initialised, i.e. you can 
     * rely on them having a valid state. However, they are arranged in an array 
     * and you don't know how the vertices are ordered in this array. 
     * createInnerCell() just receives a pointer to this array. To access the 
     * individual elements, you have to use the vertex enumerator. This functor 
     * encapsulates the vertex enumeration and is initialised by the grid 
     * properly. See the enumerator's documentation for more details.
     * 
     * If you need the position of the vertices of the cell or its size, use the 
     * enumerator.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices and fine grid cell, it is thread safe. 
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     * 
     * @param fineGridCell      Fine grid cell that you should fill with data.
     * @param fineGridVertices  Adjacent vertices of fineGridCell. To access 
     *                          elements of this array, use the enumerator.
     * @param fineGridVerticesEnumerator Enumerator for fineGridVertices.
     * @param coarseGridVertices Vertices of the next coarser level. Use the 
     *                          enumerator to access them.
     * @param coarseGridVerticesEnumerator Enumerator for coarseGridVertices.
     * @param coarseGridCell    Cell of the next coarser grid, i.e. the parent
     *                          cell within the spacetree.
     * @param fineGridPositionOfCell Position of fineGridCell within the parent 
     *                          cell. This is a d-dimensional vector where each 
     *                          entry is either 0,1, or 2.
     */
    void createCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const         fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
    );


    /**
     * Destroy a cell
     *
     * This operation is called whenever the grid decides to destroy a cell due 
     * to moving boundaries or a load rebalancing. Here, the grid does not 
     * distinguish between inner and boundary cells, i.e. if you want to react 
     * differently, you have to implement this manually.
     * 
     * Remarks:  
     * - If you need the position of the vertices of the cell or its size, use the 
     *   enumerator.
     * - If the destory is invoked due to load balancing, it is called after the 
     *   cell has been sent to another node. You can identify this case by asking
     *   cell.isRemote(). 
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices and fine grid cell, it is thread safe. 
     *
     * !!! Vertex and cell lifecycle
     * 
     * Please consult peano/grid/vertex-lifecycle.doxys for details on the 
     * vertex and cell lifecycle.
     * 
     * @param fineGridCell      Fine grid cell that you should fill with data.
     * @param fineGridVertices  Adjacent vertices of fineGridCell. To access 
     *                          elements of this array, use the enumerator.
     * @param fineGridVerticesEnumerator Enumerator for fineGridVertices.
     * @param coarseGridVertices Vertices of the next coarser level. Use the 
     *                          enumerator to access them.
     * @param coarseGridVerticesEnumerator Enumerator for coarseGridVertices.
     * @param coarseGridCell    Cell of the next coarser grid, i.e. the parent
     *                          cell within the spacetree.
     * @param fineGridPositionOfCell Position of fineGridCell within the parent 
     *                          cell. This is a d-dimensional vector where each 
     *                          entry is either 0,1, or 2.
     */
    void destroyCell(
      const peanoclaw::Cell&           fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
    );
        
        
    #ifdef Parallel
    /**
     * Merge vertex with the incoming vertex from a neighbouring computation node.
     * 
     * When Peano is running in parallel the data exchange is done vertex-wise 
     * between two grid iterations. Thus, before the touchVertexFirstTime-event
     * the vertex, sent by the computation node, which shares this vertex, is 
     * merged with the local copy of this vertex.
     *
     * !!! Heap data
     *
     * If you are working with a heap data structure, your vertices or cells, 
     * respectively, hold pointers to the heap. The received records hold 
     * pointer indices as well. However, these pointers are copies from the 
     * remote ranks, i.e. the pointers are invalid though seem to be set.
     * Receive heap data instead separately without taking the pointers in 
     * the arguments into account. 
     *
     * @param vertex    Local copy of the vertex.
     * @param neighbour Remote copy of the vertex.
     * @param fromRank  See prepareSendToNeighbour()
     * @param isForkOrJoin See preareSendToNeighbour()
     */
    void mergeWithNeighbour(
      peanoclaw::Vertex&  vertex,
      const peanoclaw::Vertex&  neighbour,
      int                                           fromRank,
      const tarch::la::Vector<DIMENSIONS,double>&   fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&   fineGridH,
      int                                           level
    );


    /**
     * Merge vertex with the incoming vertex from a neighbouring computation node.
     * 
     * When Peano is running in parallel the data exchange is done vertex-wise 
     * between two grid iterations. Thus, when the touchVertexLastTime event
     * has been called, the current vertex can be prepared for being sent to
     * the neighbouring computation node in this method. 
     *
     * @param vertex        Local vertex.
     * @param toRank        Rank of the neighbour if isForkOrJoin is unset.  
     *                      Otherwise, it is the rank of the master.
     */
    void prepareSendToNeighbour(
      peanoclaw::Vertex&  vertex,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
    );


    /**
     * Move data to neighbour
     *
     * Throughout the joins or forks of subdomains, data has to be moved 
     * between the nodes. This operation allows the user to plug into these 
     * movements. Different to the neighbour communciation, a move always 
     * implies that Peano is copying the data structure bit-wise to the 
     * remote node. However, if you have heap data and pointers, e.g., 
     * associated to your vertices/cells you have to take care yourself that 
     * this data is moved as well.
     *
     * If data is inside your computational domain, you fork, and this data 
     * now is located at the new boundary, Peano moves the data as well, i.e.
     * the move also can be a (global) copy operation.
     *
     * @param localVertex The local vertex. This is not a copy, i.e. you may 
     *                    modify the vertex before a copy of it is sent away.  
     */
    void prepareCopyToRemoteNode(
      peanoclaw::Vertex&  localVertex,
      int                                           toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
    );
    

    /**
     * @see other prepareCopyToRemoteNode() operation for the vertices.
     */
    void prepareCopyToRemoteNode(
      peanoclaw::Cell&  localCell,
      int  toRank,
      const tarch::la::Vector<DIMENSIONS,double>&   cellCentre,
      const tarch::la::Vector<DIMENSIONS,double>&   cellSize,
      int                                           level
    );


    /**
     * Merge with remote data due to fork or join
     *
     * This operation takes remote data and merges it into the local copy, as  
     * data is moved from one rank to another, e.g. Do not use an assignment 
     * operator on the whole record, as you may overwrite only PDE-specific 
     * fields in localVertex.
     *
     * @param localVertex  Local vertex data. Some information here is already 
     *                     set: Adjacency information from the master, e.g., 
     *                     already is merged. Thus, do not overwrite 
     *                     non-PDE-specific data.
     */
    void mergeWithRemoteDataDueToForkOrJoin(
      peanoclaw::Vertex&  localVertex,
      const peanoclaw::Vertex&  masterOrWorkerVertex,
      int                                       fromRank,
      const tarch::la::Vector<DIMENSIONS,double>&  x,
      const tarch::la::Vector<DIMENSIONS,double>&  h,
      int                                       level
    );


    /**
     * Merge with remote data due to fork or join
     *
     * This operation takes remote data and merges it into the local copy, as  
     * data is moved from one rank to another, e.g. Do not use an assignment 
     * operator on the whole record, as you may overwrite only PDE-specific 
     * fields in localVertex.
     *
     * @param localCell  Local cell data. Some information here is already 
     *                   set: Adjacency information from the master, e.g., 
     *                   already is merged. Thus, do not overwrite 
     *                   non-PDE-specific data.
     */
    void mergeWithRemoteDataDueToForkOrJoin(
      peanoclaw::Cell&  localCell,
      const peanoclaw::Cell&  masterOrWorkerCell,
      int                                       fromRank,
      const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
      const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
      int                                       level
    );


    /**
     * Perpare startup send to worker
     *
     * This operation is called always when we send data to a worker. It is not 
     * called when we are right in a join or fork.
     *
     * @see peano::kernel::spacetreegrid::nodes::Node::updateCellsParallelStateAfterLoad()
     */
    bool prepareSendToWorker(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
      int                                                                  worker
    );


    /**
     * Prepare send to master
     *
     * Counterpart of prepareSendToWorker() that is called at the end of each 
     * iteration if data reduction is switched on. At the moment this function 
     * is called, all the data on the local worker are already streamed to the 
     * stacks, i.e. you basically receive copies of the local cell and the 
     * local vertices. If you modify them, these changes will become undone in 
     * the subsequent iteration.
     *
     * !!! Data Consistency
     *
     * Multilevel data consistency in Peano can be tricky. Most codes thus 
     * introduce a rigorous master-owns pattern. In this case, always the 
     * vertex and cell state on the master is valid, i.e. prepareSendToMaster() 
     * informs the master about state changes. In return, prepareSendToWorker() 
     * feeds the worker with the new valid state of vertices and cells.
     * Receive from master operations thus overwrite the worker's local 
     * records with the master's data, as the master always rules.  
     */
    void prepareSendToMaster(
      peanoclaw::Cell&                       localCell,
      peanoclaw::Vertex *                    vertices,
      const peano::grid::VertexEnumerator&       verticesEnumerator,
      const peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&       coarseGridVerticesEnumerator,
      const peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&   fineGridPositionOfCell
    );


    /**
     * Merge data from the worker into the master records. This operation is 
     * called on the master, i.e. the const arguments are received copies from 
     * the worker.
     *
     * !!! Heap data
     *
     * If you are working with a heap data structure, your vertices or cells, 
     * respectively, hold pointers to the heap. The received records hold 
     * pointer indices as well. However, these pointers are copies from the 
     * remote ranks, i.e. the pointers are invalid though seem to be set.
     * Receive heap data instead separately without taking the pointers in 
     * the arguments into account.      
     */
    void mergeWithMaster(
      const peanoclaw::Cell&           workerGridCell,
      peanoclaw::Vertex * const        workerGridVertices,
      const peano::grid::VertexEnumerator& workerEnumerator,
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell,
      int                                                                  worker,
      const peanoclaw::State&          workerState,
      peanoclaw::State&                masterState
    );


    /**
     * Counterpart of prepareSendToWorker(). This operation is called once when 
     * we receive data from the master node. You can manipulate the received 
     * records - get additional data from the heap, e.g. Afterwards, the 
     * received records are merged into the worker's local data due to 
     * mergeWithWorker(). While this operation gives you access to both the 
     * cell and its adjacent vertices in the same order as prepareSendToWorker(),
     * the mergeWithWorker() operations are called on-the-fly in a (probably) 
     * different order, as their order depends on the global space-filling curve.
     *
     * If you manipulate receivedCell or receivedVertices(), respectively, 
     * these modified values will be passed to mergeWithWorker(). The classical 
     * use case is that prepareSendToWorker() sends away some heap data. This 
     * heap data then should be received in this operation, as you have all data 
     * available here in the same order as prepareSendToWorker(). Distribute the 
     * received heap data among the @f$ 2^d +1 @f$ out arguments you have 
     * available. The modified arguments then are forwarded by Peano to 
     * mergeWithWorker().
     *
     * !!! Rationale
     * 
     * The split of the receive process into two stages seems to be artificial 
     * and not very much straightforward. However, two constraints make it 
     * necessary:
     * - Some applications need a single receive point where the data received 
     *   has the same order as prepareSendToWorker().
     * - Some applications need well-initialised vertices when they descend 
     *   somewhere in the worker tree. This descend usually not in the central 
     *   element but an outer part of the tree, i.e. some vertices have to be 
     *   merged before we can also merge the cell and all other vertices.
     *
     * !!! Heap data
     *
     * If you are working with a heap data structure, your vertices or cells, 
     * respectively, hold pointers to the heap. The received records hold 
     * pointer indices as well. However, these pointers are copies from the 
     * remote ranks, i.e. the pointers are invalid though seem to be set.
     * Receive heap data instead separately without taking the pointers in 
     * the arguments into account. 
     *
     * If you receive heap data and need it in the actual merge operation, 
     * i.e. in mergeWithWorker(), we recommend to follow the subsequent steps: 
     * 
     * - Create an entry on the heap here for each vertex.
     * - Store the received data within these heap entries.
     * - Merge the heap data within mergeWithWorker().
     * - Remove the heap entries created in this operation within mergeWithWorker(). 
     */
    void receiveDataFromMaster(
      peanoclaw::Cell&                        receivedCell,
      peanoclaw::Vertex *                     receivedVertices,
      const peano::grid::VertexEnumerator&        receivedVerticesEnumerator,
      peanoclaw::Vertex * const               receivedCoarseGridVertices,
      const peano::grid::VertexEnumerator&        receivedCoarseGridVerticesEnumerator,
      peanoclaw::Cell&                        receivedCoarseGridCell,
      peanoclaw::Vertex * const               workersCoarseGridVertices,
      const peano::grid::VertexEnumerator&        workersCoarseGridVerticesEnumerator,
      peanoclaw::Cell&                        workersCoarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&    fineGridPositionOfCell
    );


    /**
     * Counterpart of mergeWithMaster()
     */
    void mergeWithWorker(
      peanoclaw::Cell&           localCell, 
      const peanoclaw::Cell&     receivedMasterCell,
      const tarch::la::Vector<DIMENSIONS,double>&  cellCentre,
      const tarch::la::Vector<DIMENSIONS,double>&  cellSize,
      int                                          level
    );


    /**
     * Counterpart of mergeWithMaster()
     */
    void mergeWithWorker(
      peanoclaw::Vertex&        localVertex,
      const peanoclaw::Vertex&  receivedMasterVertex,
      const tarch::la::Vector<DIMENSIONS,double>&   x,
      const tarch::la::Vector<DIMENSIONS,double>&   h,
      int                                           level
    );
    #endif


    /**
     * Read vertex the first time throughout one iteration
     *
     * In the Peano world, an algorithm tells the grid that the grid should be 
     * traversed. The grid then decides how to do this and runs over all cells
     * and vertices. Whenever the grid traversal reads a vertex the very first 
     * time throughout an iteration, it invokes touchVertexFirstTime() for this 
     * vertex. Then, it calls handleCell up to @f$ 2^d @f$ times for the 
     * adjacent cells and passes this vertex to these calls. Finally, the grid
     * traversal invokes touchVertexLastTime(), i.e. the counterpart of this 
     * operation.
     *
     * @image html peano/grid/geometry-vertex-inside-outside.png
     *
     * The operation is called for both each inner and boundary vertices.
     * These vertices have an attribute position, too. However, this 
     * attribute is available in Debug mode only and it should be used solely 
     * to implement assertions. To work with the vertex's position, use the 
     * attribute fineGridX instead. The latter is availble all the time (the 
     * vertex's attribute is a redundant information that is just to be used
     * for correctness and consistency checks). 
     *
     * Vertices may have persistent and non-persistent attributes (see the 
     * documentation of the DaStGen tool). Attributes that are not persistent 
     * are not stored in-between two iterations, i.e. whenever 
     * touchVertexFirstTime() is called, these attributes contain garbage. So,
     * this operation is the right place to initialise the non-persistent 
     * attributes of a vertex.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe. It is not thread safe with respect to the 
     * fine grid cell. 
     * 
     * !!! Optimisation
     * 
     * This operation is invoked if and only if the corresponding specification 
     * flag does not hold NOP. Due to this specification flag you also can define 
     * whether this operation works on the leaves only, whether it may be 
     * called in parallel by multiple threads, and whether it is fail-safe or 
     * can at least be called multiple times if a thread crashes.
     *
     * If this operation works only on leaves, the operation is sometimes not 
     * called if Peano can be sure that all adjacent cells are refined. The 
     * sometimes implies that this specification induces and optimisation - it 
     * does not enforce that the operation is not called under certain 
     * circumstances.
     *
     * @see createInnerVertex() for a description of the arguments. 
     */
    void touchVertexFirstTime(
      peanoclaw::Vertex&               fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                          fineGridH,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfVertex
    );



    /**
     * Read vertex the last time throughout one iteration
     *
     * In the Peano world, an algorithm tells the grid that the grid should be 
     * traversed. The grid then decides how to do this and runs over all cells
     * and vertices. Whenever the grid traversal reads a vertex the very first 
     * time throughout an iteration, it invokes touchVertexFirstTime() for this 
     * vertex. Then, it calls handleCell up to @f$ 2^d @f$ times for the 
     * adjacent cells and passes this vertex to these calls. Finally, the grid
     * traversal invokes touchVertexLastTime().
     *
     * @image html peano/grid/geometry-vertex-inside-outside.png
     *
     * The operation is called for both each inner and boundary vertices.
     * These vertices have an attribute position or x, too. However, this 
     * attribute is available in Debug mode only and it should be used solely 
     * to implement assertions. To work with the vertex's position, use the 
     * attribute x instead.
     *
     * Vertices may have persistent and non-persistent attributes (see the 
     * documentation of the DaStGen tool). Attributes that are not persistent 
     * are not stored in-between two iterations, i.e. whenever 
     * touchVertexFirstTime() is called, these attributes contain garbage. So,
     * this operation is the right place to do something with the non-persistent 
     * attributes. As soon as this operation terminates, these attributes are 
     * lost.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices, it is thread safe. It is not thread safe with respect to the 
     * fine grid cell. 
     * 
     * !!! Optimisation
     * 
     * This operation is invoked if and only if the corresponding specification 
     * flag does not holds NOP. Due to this specification flag you also can define 
     * whether this operation works on the leaves only, whether it may be 
     * called in parallel by multiple threads, and whether it is fail-safe or 
     * can at least be called multiple times if a thread crashes.
     *
     * If this operation works only on leaves, the operation is sometimes not 
     * called if Peano can be sure that all adjacent cells are refined. The 
     * sometimes implies that this specification induces and optimisation - it 
     * does not enforce that the operation is not called under certain 
     * circumstances.
     *
     * @see createInnerVertex() for a description of the arguments. 
     */
    void touchVertexLastTime(
      peanoclaw::Vertex&         fineGridVertex,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridX,
      const tarch::la::Vector<DIMENSIONS,double>&                    fineGridH,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                       fineGridPositionOfVertex
    );
    

    /**
     * Enter a cell
     *
     * In the Peano world, an algorithm tells the grid that the grid should be 
     * traversed. The grid then decides how to do this and runs over all cells
     * (and vertices). For each cell, it calls handleCell(), i.e. if you want 
     * your algorithm to do somethin on a cell, you should implement this 
     * operation.
     *
     * @image html peano/grid/geometry-cell-inside-outside.png
     *
     * The operation is called for each inner and boundary element and, again, 
     * you may not access the cell's adjacent vertices directly. Instead, you 
     * have to use the enumerator. For all adjacent vertices of this cell, 
     * touchVertexFirstTime() already has been called. touchVertexLastTime() has 
     * not been called yet.
     * 
     * If you need the position of the vertices of the cell or its size, use the 
     * enumerator.
     *
     * !!! Thread-safety
     *
     * This operation is not thread safe with respect to the coarse grid 
     * vertices and the coarse grid cell. With respect to the fine grid 
     * vertices and the fine grid cell, it is thread safe.
     * 
     * !!! Optimisation
     * 
     * This operation is invoked if and only if the corresponding specification 
     * flag does not hold NOP. Due to this specification flag you also can define 
     * whether this operation works on the leaves only, whether it may be 
     * called in parallel by multiple threads, and whether it is fail-safe or 
     * can at least be called multiple times if a thread crashes.
     *     
     * @see createCell() for a description of the arguments. 
     */
    void enterCell(
      peanoclaw::Cell&                 fineGridCell,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&                             fineGridPositionOfCell
    );


    /**
     * This is the counterpart of enterCell(). See this operation for a 
     * description of the arguments.
     */
    void leaveCell(
      peanoclaw::Cell&                          fineGridCell,
      peanoclaw::Vertex * const                 fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const                 coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&                          coarseGridCell,
      const tarch::la::Vector<DIMENSIONS,int>&      fineGridPositionOfCell
    );


    /**
     * Begin an iteration
     * 
     * This operation is called whenever the algorithm tells Peano that the grid 
     * is to be traversed, i.e. this operation is called before any creational 
     * mapping operation or touchVertexFirstTime() or handleCell() is called.
     * The operation receives a solver state that has to 
     * encode the solver's state. Take this attribute to set the mapping's 
     * attributes. This class' attributes will remain valid until endIteration()
     * is called. Afterwards they might contain garbage.
     *
     * !!! Parallelisation
     *
     * If you run your code in parallel, beginIteration() and endIteration() 
     * realise the following lifecycle together with the state object:
     *
     * - Receive the state from the master if there is a master.
     * - beginIteration()
     * - Distribute the state among the workers if there are workers.
     * - Merge the states from the workers (if they exist) into the own state. 
     * - endIteration()
     * - Send the state to the master if there is a master.
     *
     * @see Remesh()
     */
    void beginIteration(
      peanoclaw::State&  solverState
    );


    /**
     * Iteration is done
     * 
     * This operation is called at the very end, i.e. after all the handleCell() 
     * and toucheVertexLastTime() operations have been invoked. In this 
     * operation, you have to write all the data you will need later on back to 
     * the state object passed. Afterwards, the attributes of your mapping 
     * object (as well as global static fields) might be overwritten.  
     *
     * !!! Parallelisation
     *
     * If you run your code in parallel, beginIteration() and endIteration() 
     * realise the following lifecycle together with the state object:
     *
     * - Receive the state from the master if there is a master.
     * - beginIteration()
     * - Distribute the state among the workers if there are workers.
     * - Merge the states from the workers (if they exist) into the own state. 
     * - endIteration()
     * - Send the state to the master if there is a master.
     *
     * @see Remesh()
     */
    void endIteration(
      peanoclaw::State&  solverState
    );
    
    
    /**
     * Descend in the spacetree
     *
     * This operation is invoked right after Peano has loaded all fine grid 
     * cells but before any enterCell() is called on any of the fine grid 
     * cells, i.e. here you have access to all @f$ 3^d @f$ fine grid cells 
     * en block. The adjacent vertices on the finer grid already are loaded 
     * and touchVertexFirstTime() has been called for them.
     *
     * Hence, enterCell() and descend() are somehow redundant. However, there 
     * are cases (restriction and prolongation in multigrid methods, e.g.) 
     * where only descend() is convenient. In all other cases, it is better to 
     * use enterCell, as enterCell can be multithreaded. Also, if a fine grid 
     * cell is outside but its parent is not, descend is not invoked for this 
     * pair of cells, i.e. here you have to do something in enterCell anyway.
     *
     * To access the fine grid cells, please use the enumerator as you do for 
     * the vertices in any case.   
     *
     * !!! Thread safety
     *
     * Descend is thread-safe with respect to all arguments.
     *
     * !!! Optimisation
     * 
     * This operation is invoked if and only if the corresponding specification 
     * flag does not hold NOP. Due to this specification flag you also can define 
     * whether this operation works on the leaves only, whether it may be 
     * called in parallel by multiple threads, and whether it is fail-safe or 
     * can at least be called multiple times if a thread crashes.
     *
     * If the operation shall be called only on leaves, Peano skips the call 
     * sometimes if all fine grid cells are refined. The sometimes indicates 
     * that such a specification is an optimisation but not a valid constraint 
     * or precondition on the input arguments. 
     *     
     * @pre coarseGridCell.isInside()
     */
    void descend(
      peanoclaw::Cell * const          fineGridCells,
      peanoclaw::Vertex * const        fineGridVertices,
      const peano::grid::VertexEnumerator&                fineGridVerticesEnumerator,
      peanoclaw::Vertex * const        coarseGridVertices,
      const peano::grid::VertexEnumerator&                coarseGridVerticesEnumerator,
      peanoclaw::Cell&                 coarseGridCell
    );


    /**
     * Ascend in the spacetree
     *
     * Counterpart of descend(). Is called as soon as all leaveCell() events 
     * have terminated.
     */
    void ascend(
      peanoclaw::Cell * const    fineGridCells,
      peanoclaw::Vertex * const  fineGridVertices,
      const peano::grid::VertexEnumerator&          fineGridVerticesEnumerator,
      peanoclaw::Vertex * const  coarseGridVertices,
      const peano::grid::VertexEnumerator&          coarseGridVerticesEnumerator,
      peanoclaw::Cell&           coarseGridCell
    );

};
#endif
