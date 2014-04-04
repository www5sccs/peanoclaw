#ifndef _BATHYMETRYHELPER_H_
#define _BATHYMETRYHELPER_H_

#include <iostream>

#include <netcdf.h>

#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"

class BathymetryHelper {
    public:
        BathymetryHelper(const std::string& nc_filename,
                         const std::string& lon_name,
                         const std::string& lat_name,
                         const std::string& dataset_name
        );

        tarch::la::Vector<DIMENSIONS,size_t> getClosestInterval1D(float probe, float *data, const size_t size);

        float getHeight(double longitude, double latitude);

        virtual ~BathymetryHelper();
    private:
        size_t lat_length;
        size_t lon_length;

        float* latitude_values;
        float* longitude_values;

        int ncid;
        int data_varid;
};

#endif // _BATHYMETRYHELPER_H_
