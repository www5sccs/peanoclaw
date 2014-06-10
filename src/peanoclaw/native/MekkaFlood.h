#ifndef __MEKKAFLOOD_H__
#define __MEKKAFLOOD_H__

#if defined(SWE) || defined(PEANOCLAW_FULLSWOF2D)

#include <cstdio>
//#include <png.h>

#include "tarch/la/Vector.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/native/SWEKernel.h"
#include "peanoclaw/native/scenarios/SWEScenario.h"

//#include "BathymetryHelper.h"

#include "peanoclaw/native/dem.h"

namespace peanoclaw {
  namespace native {
    class MekkaFlood_SWEKernelScenario;
  }
}

// mekka coordinates
//const float mekka_lat = 21.4167f;
//const float mekka_lon = 39.8167f;

// jeddah coordinates
const float mekka_lat = 21.5f;
const float mekka_lon = 39.0f;


class peanoclaw::native::MekkaFlood_SWEKernelScenario : public peanoclaw::native::scenarios::SWEScenario {
    public:
        //MekkaFlood_SWEKernelScenario(double domainSize);
        MekkaFlood_SWEKernelScenario(
          DEM& dem,
          const tarch::la::Vector<DIMENSIONS,int>&    subdivisionFactor,
          const tarch::la::Vector<DIMENSIONS,double>& minimalMeshWidth,
          const tarch::la::Vector<DIMENSIONS,double>& maximalMeshWidth,
          double                                      globalTimestepSize,
          double                                      endTime
        );

        ~MekkaFlood_SWEKernelScenario();

        virtual void initializePatch(peanoclaw::Patch& patch);
        virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing);
        virtual void update(peanoclaw::Patch& patch);
 
        int scale;

        tarch::la::Vector<DIMENSIONS,double> getDomainOffset() const;
        tarch::la::Vector<DIMENSIONS,double> getDomainSize() const;
        tarch::la::Vector<DIMENSIONS,double> getInitialMinimalMeshWidth() const;
        tarch::la::Vector<DIMENSIONS,int>    getSubdivisionFactor() const;

        double getGlobalTimestepSize() const;

        double getEndTime() const;

        double getInitialTimestepSize() const;

    private:
        tarch::la::Vector<DIMENSIONS, double> mapCoordinatesToMesh(double longitude, double latitude);
        tarch::la::Vector<DIMENSIONS, double> mapMeshToCoordinates(double x, double y);
        double mapMeshToMap(tarch::la::Vector<DIMENSIONS, double>& coords);

        tarch::la::Vector<DIMENSIONS, int>    _subdivisionFactor;
        tarch::la::Vector<DIMENSIONS, double> _domainSize;
        tarch::la::Vector<DIMENSIONS, double> _domainOffset;
        tarch::la::Vector<DIMENSIONS, double> _minialMeshWidth;
        tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;
        double                                _globalTimestepSize;
        double                                _endTime;

        DEM& dem;
        //BathymetryHelper bathymetryHelper;
        
#if 0
        png_image mekka_map; /* The control structure used by libpng */
        uint8_t* mekka_map_data;
#endif
};
#endif

#endif // __MEKKAFLOOD_H__
