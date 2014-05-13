#ifndef PEANOCLAW_GRID_SUBGRIDLEVELCONTAINER
#define PEANOCLAW_GRID_SUBGRIDLEVELCONTAINER

#include "peanoclaw/Patch.h"

namespace peanoclaw {
  namespace grid {
    class SubgridLevelContainer;
  }
}

class peanoclaw::grid::SubgridLevelContainer {

  public:
    typedef peanoclaw::Patch Level[FIVE_POWER_D];

  private:
    const static int maximumNumberOfLevels = 100;

    int _index;
    peanoclaw::Patch _firstLevelSubgrid;
    peanoclaw::Patch _subgrids[FIVE_POWER_D * maximumNumberOfLevels];

  public:
    SubgridLevelContainer();

    /**
     * Sets the subgrid on the first level on the according
     * cell and vertices.
     */
    void setFirstLevel(
      peanoclaw::Cell&                     cell,
      peanoclaw::Vertex * const            vertices,
      const peano::grid::VertexEnumerator& verticesEnumerator
    );

    /**
     * Adds a new level of subgrids depending on the cells and vertices.
     */
    void addNewLevel(
      peanoclaw::Cell * const fineGridCells,
      peanoclaw::Vertex * const fineGridVertices,
      const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
    );

    /**
     * Removes the current level and deletes the according Subgrids.
     */
    void removeCurrentLevel();

    /**
     * Returns the 5^d subgrids of the current level.
     */
//    Level& getCurrentLevelSubgrids() const;

    /**
     * Returns the index of the current level.
     */
    int getCurrentLevel() const;
};

#endif //PEANOCLAW_GRID_SUBGRIDLEVELCONTAINER
