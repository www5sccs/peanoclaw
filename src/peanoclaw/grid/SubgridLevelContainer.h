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

    /**
     * The size of a top level. Here we have to consider
     * 3^d subgrids, since a top level may also refer
     * to a root cell of a worker that has subgrids
     * surrounding the top level cell.
     */
    const static int TopLevel = 3;

    /**
     * The size of a normal full level. Here we have to
     * consider 5^d subgrids.
     */
    const static int Full = 5;

    int _index;
//    peanoclaw::Patch _firstLevelSubgrid;
    peanoclaw::Patch _subgrids[FIVE_POWER_D * maximumNumberOfLevels];

    /**
     * Creates and sets the subgrids for a specific level.
     *
     * The difference to method addNewLevel is that in this
     * method 5x5 arrays of subgrids can be processed as
     * well as 3x3 arrays. Hence, this method can be used for
     * the addNewLevel method and for the setFirstLevel method.
     */
    void createAndSetSubgridsForLevel(
      peanoclaw::Cell * const cells,
      peanoclaw::Vertex * const vertices,
      const peano::grid::VertexEnumerator& enumerator,
      int subgridArraySize
    );

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
      peanoclaw::Cell * const cells,
      peanoclaw::Vertex * const vertices,
      const peano::grid::VertexEnumerator& enumerator
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
