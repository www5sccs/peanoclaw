/*
 * TestEnumerator.h
 *
 *  Created on: Jul 5, 2011
 *      Author: unterweg
 */

#ifndef _PEANO_APPLICATIONS_PEANOCLAW_TESTS_TESTENUMERATOR_H_
#define _PEANO_APPLICATIONS_PEANOCLAW_TESTS_TESTENUMERATOR_H_

#include "peano/kernel/gridinterface/VertexEnumerator.h"
#include <sstream>

namespace peano {
  namespace applications {
    namespace peanoclaw {
      namespace tests {
        class TestVertexEnumerator;
      }
    }
  }
}

/**
 * This is a stub for tests in the peanoclaw component. Only the methods that are needed are
 * implemented in the .cpp-file. Feel free to implement further methods appropriately, if you need
 * them for test cases.
 * This Enumerator implements an identity mapping.
 * @author Kristof Unterweger, Michael Lieb
 */
class peano::applications::peanoclaw::tests::TestVertexEnumerator
  : public peano::kernel::gridinterface::VertexEnumerator
{
  private:
    /**
     * The cellsize to be returned by the TestVertexEnumerator.
     */
    Vector _cellSize;

  public:
    /**
     * @param indexMapping This array of integers defines the mapping from
     * the local indices {0..NUMBER_OF_VERTICES_PER_ELEMENT} to the global indices.
     * @param cellSize The cellsize to be returned by the TestVertexEnumerator.
     */
    TestVertexEnumerator( const Vector cellSize );
    virtual ~TestVertexEnumerator();

    /**
     * Sets the current cell size.
     */
    void setCellSize(const Vector cellSize);

    // Methods from VertexEnumerator
    int                                     operator() (int localVertexNumber) const;
    int                                     operator() (const LocalVertexIntegerIndex& localVertexNumber ) const;
    int                                     operator() (const LocalVertexBitsetIndex& localVertexNumber ) const { throw ""; }
    Vector                                  getVertexPosition(int localVertexNumber) const { return Vector(-1); };
    Vector                                  getVertexPosition(const LocalVertexIntegerIndex& localVertexNumber) const { return Vector(-1); };
    Vector                                  getVertexPosition(const LocalVertexBitsetIndex& localVertexNumber) const { return Vector(-1); };
    Vector                                  getVertexPosition() const { return Vector(-1); };
    Vector                                  getCellCenter() const {return Vector(-2.0);};
    int                                     getLevel() const {return -1;};
    peano::kernel::gridinterface::CellFlags getCellFlags() const;
    Vector                                  getCellSize() const;
    bool                                    isStationarySubdomain() const {return false;}
    bool                                    subdomainContainsParallelBoundary() const {return true;}
    int                                     cell(const LocalVertexIntegerIndex& localVertexNumber) const { return 0; }
    std::string                             toString() const {
      std::stringstream ss;
      ss << "This is a TestEnumerator." <<std::endl;
      return ss.str();
      };
};

#endif /* _PEANO_APPLICATIONS_PEANOCLAW_TESTS_TESTENUMERATOR_H_ */
