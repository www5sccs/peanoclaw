#ifndef __MEKKAFLOOD_H__
#define __MEKKAFLOOD_H__

#if defined(SWE) || defined(PEANOCLAW_FULLSWOF2D)

#include <cstdio>
//#include <png.h>

#include "tarch/la/Vector.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/native/SWEKernel.h"

//#include "BathymetryHelper.h"

#include "peanoclaw/native/dem.h"

// mekka coordinates
//const float mekka_lat = 21.4167f;
//const float mekka_lon = 39.8167f;

// jeddah coordinates
const float mekka_lat = 21.5f;
const float mekka_lon = 39.0f;


class MekkaFlood_SWEKernelScenario : public peanoclaw::native::SWEKernelScenario {
    public:
        //MekkaFlood_SWEKernelScenario(double domainSize);
        MekkaFlood_SWEKernelScenario(DEM& dem);

        ~MekkaFlood_SWEKernelScenario();

        virtual void initializePatch(peanoclaw::Patch& patch);
        virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing);
        virtual void update(peanoclaw::Patch& patch);
 
        int scale;

    private:
        tarch::la::Vector<DIMENSIONS, double> mapCoordinatesToMesh(double longitude, double latitude);
        tarch::la::Vector<DIMENSIONS, double> mapMeshToCoordinates(double x, double y);
        double mapMeshToMap(tarch::la::Vector<DIMENSIONS, double>& coords);

        //double domainSize;

        DEM& dem;
        //BathymetryHelper bathymetryHelper;
        
#if 0
        png_image mekka_map; /* The control structure used by libpng */
        uint8_t* mekka_map_data;
#endif
};
#endif

#endif // __MEKKAFLOOD_H__
