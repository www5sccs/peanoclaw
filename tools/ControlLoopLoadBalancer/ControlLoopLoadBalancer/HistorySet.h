#ifndef _CONTROL_LOOP_LOAD_BALANCER_HISTORYSET_H_
#define _CONTROL_LOOP_LOAD_BALANCER_HISTORYSET_H_

#include <map>

#include "ControlLoopLoadBalancer/History.h"

namespace mpibalancing {
    namespace ControlLoopLoadBalancer {
        template <typename Data>
        class HistorySetReduction;

        template <typename Key, typename Data>
        class HistorySet;
        
        template <typename Key, typename Data, typename _Entry>
        class StdHistoryMap;
    }
}

template <typename Data>
class mpibalancing::ControlLoopLoadBalancer::HistorySetReduction {
    public:
        virtual ~HistorySetReduction() {}
        virtual void evaluate(const History<Data>& data)  = 0;
    protected:
        HistorySetReduction() {}

};

template <typename Key, typename Data>
class mpibalancing::ControlLoopLoadBalancer::HistorySet {
    public:
        virtual ~HistorySet() { }
        virtual int size() const = 0;
  
        // This method iterates over all available histories and calls the specified 
        // reduction operation for each one of them.
        virtual void reduce(HistorySetReduction<Data>& operation, bool reversed=false) const = 0;

        // This method returns the history which is associated with the given key.
        virtual History<Data>& getHistory(Key id) = 0;
        virtual void deleteHistory(Key id) = 0;

        virtual void advanceInTime() = 0;

        virtual void reset() = 0;
    protected:
        HistorySet() {}
};

template <typename Key, typename Data, typename _Entry = mpibalancing::ControlLoopLoadBalancer::History<Data> >
class mpibalancing::ControlLoopLoadBalancer::StdHistoryMap : public HistorySet<Key, Data>  {
    public:
        StdHistoryMap() {}
        virtual ~StdHistoryMap() { }
 
        virtual int size() const { return _historySet.size(); }
 
        virtual void reduce(HistorySetReduction<Data>& operation, bool reversed=false) const {
            if (reversed) {
                typename std::map<Key,_Entry>::const_reverse_iterator it = _historySet.rbegin();
                while (it != _historySet.rend()) {
                    operation.evaluate(it->second);
                    it++;
                }
            } else {
                typename std::map<Key,_Entry>::const_iterator it = _historySet.begin();
                while (it != _historySet.end()) {
                    operation.evaluate(it->second);
                    it++;
                }
            }
        }

        virtual History<Data>& getHistory(Key id) {
            return _historySet[id];
        }

        virtual void deleteHistory(Key id) {
            _historySet.erase(id);
        }

        virtual void advanceInTime() {
            typename std::map<Key,_Entry>::iterator it = _historySet.begin();
            while (it != _historySet.end()) {
                it->second.advanceInTime();
                it++;
            }
        }

        virtual void reset() {
            _historySet.clear();
        }

    private:
        std::map<Key,_Entry> _historySet;
};

#endif // _CONTROL_LOOP_LOAD_BALANCER_STATE_H_
