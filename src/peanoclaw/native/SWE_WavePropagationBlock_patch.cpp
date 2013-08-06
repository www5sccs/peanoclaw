#include "SWE_WavePropagationBlock_patch.hh"

#include <cassert>
#include <string>
#include <limits>

SWE_WavePropagationBlock_patch::SWE_WavePropagationBlock_patch(peanoclaw::Patch& patch)
  : _patch(patch),
    SWE_WavePropagationBlock(
          patch.getSubdivisionFactor()(0),
          patch.getSubdivisionFactor()(1),
          patch.getSubcellSize()(0),
          patch.getSubcellSize()(1)
      )
{
    setBoundaryType(BND_LEFT, CONNECT);
    setBoundaryType(BND_RIGHT, CONNECT);
    setBoundaryType(BND_BOTTOM, CONNECT);
    setBoundaryType(BND_TOP, CONNECT);
    setBoundaryConditions();

    tarch::la::Vector<DIMENSIONS,int> subcellIndex;

    tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = _patch.getSubdivisionFactor();

    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::h[x+1][y+1] = _patch.getValueUOld(subcellIndex, 0);
        }
    }
     
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::hu[x+1][y+1] = _patch.getValueUOld(subcellIndex, 1);
        }
    }
  
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            SWE_Block::hv[x+1][y+1] = _patch.getValueUOld(subcellIndex, 2);
        }
    }
 
    for (int x = -1; x < subdivisionFactor(0)+1; x++) {
        for (int y = -1; y < subdivisionFactor(1)+1; y++) {
            SWE_Block::b[x+1][y+1] = 0.0;
        }
    }
}

SWE_WavePropagationBlock_patch::~SWE_WavePropagationBlock_patch() 
{
    tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = _patch.getSubdivisionFactor();

    tarch::la::Vector<DIMENSIONS,int> subcellIndex;
    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            _patch.setValueUNew(subcellIndex, 0, SWE_Block::h[x+1][y+1]);

        }
    }

    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            _patch.setValueUNew(subcellIndex, 1, SWE_Block::hu[x+1][y+1]);

        }
    }

    for (int x = 0; x < subdivisionFactor(0); x++) {
        for (int y = 0; y < subdivisionFactor(1); y++) {
            subcellIndex(0) = x;
            subcellIndex(1) = y;
            _patch.setValueUNew(subcellIndex, 2, SWE_Block::hv[x+1][y+1]);
        }
    }
}
