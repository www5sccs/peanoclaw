#ifndef _CONTROL_LOOP_LOAD_BALANCER_WORKERDATA_H_
#define _CONTROL_LOOP_LOAD_BALANCER_WORKERDATA_H_

#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"

#include <ostream>

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        class WorkerData;
    }
}

//This class contains all information provided by the workers which is sent in an bottom-up fashion.
//Nevertheless, we will (mis-)use this class to store information provided by the master as well,
//which does only provide partial data. However, this might change in the near future.
class  mpibalancing::ControlLoopLoadBalancer::WorkerData {
    public:
        WorkerData();
        virtual ~WorkerData();
 
        const int getRank() const;
        void setRank(int rank);
  
        const tarch::la::Vector<DIMENSIONS,double>& getBoundingBoxOffset() const;
        void setBoundingBoxOffset(const tarch::la::Vector<DIMENSIONS,double>& offset);

        const tarch::la::Vector<DIMENSIONS,double>& getBoundingBoxSize() const;
        void setBoundingBoxSize(const tarch::la::Vector<DIMENSIONS,double>& size);
       
        const bool isForkAllowed() const;
        void setForkAllowed(bool flag);
        const bool isJoinAllowed() const;
        void setJoinAllowed(bool flag);

        // Due to constraints or certain properties of the current (global) grid, 
        // it may not be possible to execute a given load balancing command.
        // Therefore, we store both the desired load balancing command and 
        // the actual loadbalancing command. This allows us to re-evaluate 
        // and perhaps retry a previous load balancing attempt.
        // One typical example is an Erase Operation which is not possible 
        // due to decomposition. However, a Join might not be possible as
        // this might degenerate the grid badly. In this particular case 
        // the desired load balancing command would be Join, but the actual 
        // load balancing command is Continue.
        const int getDesiredLoadBalancingCommand() const;
        void setDesiredLoadBalancingCommand(int command);
        const int getActualLoadBalancingCommand() const;
        void setActualLoadBalancingCommand(int command);

        const double getWaitedTime() const;
        void setWaitedTime(double time);

        const double getNumberOfInnerVertices() const;
        void setNumberOfInnerVertices(double vertices);

        const double getNumberOfBoundaryVertices() const;
        void setNumberOfBoundaryVertices(double vertices);

        const double getNumberOfOuterVertices() const;
        void setNumberOfOuterVertices(double vertices);

        const double getNumberOfInnerCells() const;
        void setNumberOfInnerCells(double cells);

        const double getNumberOfOuterCells() const;
        void setNumberOfOuterCells(double cells);

        const int getMaxLevel() const;
        void setMaxLevel(int level);

        const int getCurrentLevel() const;
        void setCurrentLevel(int level);

        const double getLocalWorkload() const;
        void setLocalWorkload(double workload);

        const double getTotalWorkload() const;
        void setTotalWorkload(double workload);
 
        const double getMaxWorkload() const;
        void setMaxWorkload(double workload);
 
        const double getMinWorkload() const;
        void setMinWorkload(double workload);

        const double getParentCellLocalWorkload() const;
        void setParentCellLocalWorkload(double workload);

        const bool getCouldNotEraseDueToDecompositionFlag() const;
        void setCouldNotEraseDueToDecompositionFlag(bool flag);

        // Use this function to reset all available data.
        void reset();

        // define a comparison operator to enable the use of sets
        bool operator<(const WorkerData& right); 
     private:
        int _rank;
        tarch::la::Vector<DIMENSIONS,double> _boundingBoxOffset;
        tarch::la::Vector<DIMENSIONS,double> _boundingBoxSize;
        bool _joinIsAllowed;
        bool _forkIsAllowed;

        int _desiredLoadBalancingCommand;
        int _actualLoadBalancingCommand;

        double  _waitedTime;
        double  _numberOfInnerVertices;
        double  _numberOfBoundaryVertices;
        double  _numberOfOuterVertices;
        double  _numberOfInnerCells;
        double  _numberOfOuterCells;
        int     _maxLevel;
        int     _currentLevel;
        double  _localWorkload;
        double  _totalWorkload;
        double  _maxWorkload;
        double  _minWorkload;
        double  _parentCellLocalWorkload;
        bool    _couldNotEraseDueToDecomposition;
};

std::ostream& operator<<(std::ostream& stream, const mpibalancing::ControlLoopLoadBalancer::WorkerData& workerData);

#endif // _CONTROL_LOOP_LOAD_BALANCER_WORKERDATA_H_
