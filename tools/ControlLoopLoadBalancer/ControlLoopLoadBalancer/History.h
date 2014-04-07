#ifndef _CONTROL_LOOP_LOAD_BALANCER_HISTORY_H_
#define _CONTROL_LOOP_LOAD_BALANCER_HISTORY_H_

#include "tarch/la/Vector.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        template <typename Data>
        class History;

        template <typename Data, unsigned int K>
        class FIFOHistory;
    }
}

template <typename Data>
class mpibalancing::ControlLoopLoadBalancer::History {
    public:
        // Just the basic destructor code line as usual.
        virtual ~History(void) {}

        // this number determines how many versions we may keep for evaluation.
        virtual int size(void) const = 0;

        // Use this method to get older versions of our tracked data values.
        // 0: Current version
        // 1: previous version
        // 2: previous version of previous version
        // 3: ...
        // ...
        // size-1: oldest known version
        virtual const Data& getPastItem(int id) const = 0;

        // Get the current item.
        // Note: While we could use getPastItem() here as well, 
        // we explicitly allow changes to the current item as we still have the opportunity 
        // to change history :)
        virtual Data& getCurrentItem(void) = 0;

        // Make history! No more changes will be allowed for the current data version 
        // and we move forward in time, enabling changes to a new version.
        virtual void advanceInTime(void) = 0;

        virtual void reset(void) = 0;
    protected:
        // We encourage inheritence of this class instead of using it directly.
        History(void) {}
};

// Implement a classic FIFO style history.
template <typename Data, unsigned int K>
class mpibalancing::ControlLoopLoadBalancer::FIFOHistory : public History<Data> {
    public:
        FIFOHistory() { _offset = 0;  }

        virtual int size() const { return K; }
        virtual const Data& getPastItem(int id) const {
            return _history(translatePastID(id));
        }

        virtual Data& getCurrentItem(void) {
            return _history(_offset);
        }
 
        virtual void advanceInTime(void) {
            _offset = (_offset + 1) % size();
            _history(_offset) = Data(); // reset content
        }

        virtual void reset(void) {
            for (int i=0; i < K; i++) {
                _history[i] = Data();
            }
        }

    private:
        int translatePastID(int id) const { 
            return (_offset+size()-id) % size(); 
        }

        int _offset;
        tarch::la::Vector<K,Data> _history;
};

#endif //  _CONTROL_LOOP_LOAD_BALANCER_HISTORY_H_
