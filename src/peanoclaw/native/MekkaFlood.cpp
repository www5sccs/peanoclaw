#if defined(SWE) || defined(PEANOCLAW_FULLSWOF2D)

#include <algorithm>

#include <png.h>

#include "peanoclaw/native/MekkaFlood.h"

//#define BREAKINGDAMTEST

/*MekkaFlood_SWEKernelScenario::MekkaFlood_SWEKernelScenario(double domainSize) : 
    domainSize(domainSize),
    bathymetryHelper("smith_sandwell.nc", "X2MIN", "Y2MIN","ROSE")
{

}*/

MekkaFlood_SWEKernelScenario::MekkaFlood_SWEKernelScenario(DEM& dem) : dem(dem)
{
 
    scale = 2.0;

#if 0
    // open png with mekka_map
    memset(&mekka_map, 0, (sizeof mekka_map));
    mekka_map.version = PNG_IMAGE_VERSION;
 
    /* The first argument is the file to read: */
    if (png_image_begin_read_from_file(&mekka_map, "tex2d_cropped.png")) {
         /* Set the format in which to read the PNG file; this code chooses a
          * simple sRGB format with a non-associated alpha channel, adequate to
          * store most images.
          */
         //mekka_map.format = PNG_FORMAT_RGBA;
         mekka_map.format = PNG_FORMAT_GRAY; // we only need gray scale

         /* Now allocate enough memory to hold the image in this format; the
          * PNG_IMAGE_SIZE macro uses the information about the image (width,
          * height and format) stored in 'image'.
          */
         std::cout << "image buffer size: " << PNG_IMAGE_SIZE(mekka_map) << std::endl;
         mekka_map_data = static_cast<uint8_t*>(malloc(PNG_IMAGE_SIZE(mekka_map)));

         if (mekka_map_data != NULL &&
            png_image_finish_read(&mekka_map, NULL/*background*/, mekka_map_data,
               0/*row_stride*/, NULL/*colormap*/)) {
         
              std::cout << "got png image data: width=" << mekka_map.width << ", height=" << mekka_map.height << std::endl;
         }

    } else {

        /* Something went wrong reading or writing the image.  libpng stores a
         * textual message in the 'png_image' structure:
         */
         std::cerr << "pngtopng: error: " << mekka_map.message << std::endl;
    }
#endif
    
}

MekkaFlood_SWEKernelScenario::~MekkaFlood_SWEKernelScenario() {
    // cleanup png data
#if 0
    if (mekka_map_data == NULL)
       png_image_free(&mekka_map);

    else
       free(mekka_map_data);
#endif
}

static double bilinear_interpolate(double x00, double y00,
                                   double x11, double y11,
                                   double x, double y,
                                   double f00, double f10, double f01, double f11) 
{
    // NOTE: dataset is organized as latitude / longitude

    double width = x11 - x00;
    double height = y11 - y00;

    // 1D interpolation along bottom
    double bottom_interpolated = ((x11 - x) * f00) / width + 
                                 ((x - x00) * f10) / width;

    // 1D interpolation along top
    double top_interpolated = ((x11 - x) * f01) / width + 
                              ((x - x00) * f11) / width;

    // complete the bilinear interpolation along latitude
    double data_value = ((y11 - y) * bottom_interpolated) / height + 
                        ((y - y00) * top_interpolated) / height;
 
    return data_value;
}

static double interpolation_error(peanoclaw::Patch& patch, int unknown) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = patch.getSubdivisionFactor();
 
    // interpolatation error inside patch
    double max_error = 0.0;
  
    subcellIndex(0) = 0;
    subcellIndex(1) = 0;
    double f00 = patch.getValueUNew(subcellIndex, 0);
 
    subcellIndex(0) = subdivisionFactor(0)-1;
    subcellIndex(1) = 0;
    double f10 = patch.getValueUNew(subcellIndex, 0);
 
    subcellIndex(0) = 0;
    subcellIndex(1) = subdivisionFactor(1)-1;
    double f01 = patch.getValueUNew(subcellIndex, 0);
 
    subcellIndex(0) = subdivisionFactor(0)-1;
    subcellIndex(1) = subdivisionFactor(1)-1;
    double f11 = patch.getValueUNew(subcellIndex, 0);

    for (int yi = 0; yi < subdivisionFactor(1); yi++) {
        for (int xi = 0; xi < subdivisionFactor(0); xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
            double value = patch.getValueUNew(subcellIndex, 0);
 
            double interpolated_value = bilinear_interpolate(0.0, 0.0,
                                                             patchSize(0), patchSize(1),
                                                             xi*meshWidth(0), yi*meshWidth(0),
                                                             f00, f10, f01, f11);


            double error = std::abs(value - interpolated_value);

            max_error = std::max(max_error, error);
        }
    }

 
    // interpolatation error with ghostlayer
    double max_error_ghost = 0.0;
  
    subcellIndex(0) = -1;
    subcellIndex(1) = -1;
    f00 = patch.getValueUOld(subcellIndex, 0);
 
    subcellIndex(0) = subdivisionFactor(0);
    subcellIndex(1) = -1;
    f10 = patch.getValueUOld(subcellIndex, 0);
 
    subcellIndex(0) = -1;
    subcellIndex(1) = subdivisionFactor(1);
    f01 = patch.getValueUOld(subcellIndex, 0);
 
    subcellIndex(0) = subdivisionFactor(0);
    subcellIndex(1) = subdivisionFactor(1);
    f11 = patch.getValueUOld(subcellIndex, 0);

    for (int yi = -1; yi < subdivisionFactor(1)+1; yi++) {
        for (int xi = -1; xi < subdivisionFactor(0)+1; xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
            double value = patch.getValueUOld(subcellIndex, 0);
 
            double interpolated_value = bilinear_interpolate(-meshWidth(0), -meshWidth(1),
                                                             patchSize(0)+meshWidth(0), patchSize(1)+meshWidth(1),
                                                             xi*meshWidth(0), yi*meshWidth(1),
                                                             f00, f10, f01, f11);


            double error = std::abs(value - interpolated_value);

            max_error_ghost = std::max(max_error, error);
        }
    }

    return std::max(max_error,max_error_ghost);
}


static double interpolation_error_coarse_fine_gradient(peanoclaw::Patch& patch, int unknown) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> coarseSubcellIndex;

    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = patch.getSubdivisionFactor();
 
    double max_error = 0.0;
 
    tarch::la::Vector<DIMENSIONS, double> meshPos;
    for (int yi = 1; yi < subdivisionFactor(1)-1; yi++) {
        for (int xi = 1; xi < subdivisionFactor(0)-1; xi++) {
            // coarse info ------------------------------
            coarseSubcellIndex(0) = std::floor(xi / 3.0) * 3;
            coarseSubcellIndex(1) = std::floor(yi / 3.0) * 3;
            tarch::la::Vector<DIMENSIONS, double> meshPos_00 = patch.getSubcellPosition(coarseSubcellIndex);
            double f00 = patch.getValueUNew(coarseSubcellIndex, unknown);
         
            coarseSubcellIndex(0) = std::ceil(xi / 3.0) * 3;
            coarseSubcellIndex(1) = std::floor(yi / 3.0) * 3;
            double f10 = patch.getValueUNew(coarseSubcellIndex, unknown);
         
            coarseSubcellIndex(0) = std::floor(xi / 3.0) * 3;
            coarseSubcellIndex(1) = std::ceil(yi / 3.0) * 3;
            double f01 = patch.getValueUNew(coarseSubcellIndex, unknown);
         
            coarseSubcellIndex(0) = std::ceil(xi / 3.0) * 3;
            coarseSubcellIndex(1) = std::ceil(yi / 3.0) * 3;
            tarch::la::Vector<DIMENSIONS, double> meshPos_11 = patch.getSubcellPosition(coarseSubcellIndex);
            double f11 = patch.getValueUNew(coarseSubcellIndex, unknown);

            // fine info -------------------------------
            // center
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_11 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_11 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_11 = value_11 - interpolated_value_11;
            
            // left
            subcellIndex(0) = xi-1;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_01 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_01 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_01 = value_01 - interpolated_value_01;

            // right
            subcellIndex(0) = xi+1;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_21 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_21 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_21 = value_21 - interpolated_value_21;

            // bottom
            subcellIndex(0) = xi;
            subcellIndex(1) = yi-1;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_10 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_10 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_10 = value_10 - interpolated_value_10;
 
            // top
            subcellIndex(0) = xi;
            subcellIndex(1) = yi+1;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_12 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_12 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_12 = value_12 - interpolated_value_12;


            double left_gradient = std::abs(error_11 - error_01);
            double right_gradient = std::abs(error_21 - error_11);

            double bottom_gradient = std::abs(error_11 - error_10);
            double top_gradient = std::abs(error_12 - error_11);


            double max_gradient_x = std::max(left_gradient, right_gradient) / meshWidth(0);
            double max_gradient_y = std::max(left_gradient, right_gradient) / meshWidth(1);

            max_error = std::max(max_error, max_gradient_x);
            max_error = std::max(max_error, max_gradient_y);
        }
    }

    return max_error;
}


static double interpolation_error_gradient(peanoclaw::Patch& patch, int unknown) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = patch.getSubdivisionFactor();
 
    // interpolatation error inside patch
    double max_error = 0.0;
  
    subcellIndex(0) = 0;
    subcellIndex(1) = 0;
    tarch::la::Vector<DIMENSIONS, double> meshPos_00 = patch.getSubcellPosition(subcellIndex);
    double f00 = patch.getValueUNew(subcellIndex, unknown);
 
    subcellIndex(0) = subdivisionFactor(0)-1;
    subcellIndex(1) = 0;
    double f10 = patch.getValueUNew(subcellIndex, unknown);
 
    subcellIndex(0) = 0;
    subcellIndex(1) = subdivisionFactor(1)-1;
    double f01 = patch.getValueUNew(subcellIndex, unknown);
 
    subcellIndex(0) = subdivisionFactor(0)-1;
    subcellIndex(1) = subdivisionFactor(1)-1;
    tarch::la::Vector<DIMENSIONS, double> meshPos_11 = patch.getSubcellPosition(subcellIndex);
    double f11 = patch.getValueUNew(subcellIndex, unknown);
 
    tarch::la::Vector<DIMENSIONS, double> meshPos;
    for (int yi = 1; yi < subdivisionFactor(1)-1; yi++) {
        for (int xi = 1; xi < subdivisionFactor(0)-1; xi++) {
            // center
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_11 = patch.getValueUNew(subcellIndex, unknown);
            double interpolated_value_11 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_11 = value_11 - interpolated_value_11;
            
            // left
            subcellIndex(0) = xi-1;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_01 = patch.getValueUNew(subcellIndex, unknown);
            double interpolated_value_01 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_01 = value_01 - interpolated_value_01;

            // right
            subcellIndex(0) = xi+1;
            subcellIndex(1) = yi;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_21 = patch.getValueUNew(subcellIndex, unknown);
            double interpolated_value_21 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_21 = value_21 - interpolated_value_21;

            // bottom
            subcellIndex(0) = xi;
            subcellIndex(1) = yi-1;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_10 = patch.getValueUNew(subcellIndex, unknown);
            double interpolated_value_10 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_10 = value_10 - interpolated_value_10;
 
            // top
            subcellIndex(0) = xi;
            subcellIndex(1) = yi+1;
            meshPos = patch.getSubcellPosition(subcellIndex);
            double value_12 = patch.getValueUNew(subcellIndex, 0);
            double interpolated_value_12 = bilinear_interpolate(meshPos_00(0), meshPos_00(1),
                                                             meshPos_11(0), meshPos_11(1),
                                                             meshPos(0), meshPos(1),
                                                             f00, f10, f01, f11);
            double error_12 = value_12 - interpolated_value_12;


            double left_gradient = std::abs(error_11 - error_01);
            double right_gradient = std::abs(error_21 - error_11);

            double bottom_gradient = std::abs(error_11 - error_10);
            double top_gradient = std::abs(error_12 - error_11);


            double max_gradient_x = std::max(left_gradient, right_gradient) / meshWidth(0);
            double max_gradient_y = std::max(left_gradient, right_gradient) / meshWidth(1);

            max_error = std::max(max_error, max_gradient_x);
            max_error = std::max(max_error, max_gradient_y);
        }
    }

    return max_error;
}

void MekkaFlood_SWEKernelScenario::initializePatch(peanoclaw::Patch& patch) {
    // dam coordinates
    //double x0=domainSize*0.5;
    //double y0=domainSize*0.5;
  
    double x_size = (dem.upper_right(0) - dem.lower_left(0)) / scale;
    double y_size = (dem.upper_right(1) - dem.lower_left(1))  / scale;

    double x0=(x_size) * 0.5;
    double y0=(y_size) * 0.5;
    
    // Riemann states of the dam break problem
    double radDam = 0.05*std::min(x_size,y_size);
#if defined(BREAKINGDAMTEST)
    double hl = 2.;
#else
    double hl = 0.0;
#endif
    double ul = 0.;
    double vl = 0.;
    double hr = 0.; // 1
    double ur = 0.;
    double vr = 0.;

    double q0 = 0;
    double q1 = 0;
    
    // compute from mesh data
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
 
    int ghostlayerWidth = patch.getGhostlayerWidth();

    // initialize new part only
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    tarch::la::Vector<DIMENSIONS, double> meshPos;
    for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
        for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
 
            meshPos = patch.getSubcellPosition(subcellIndex);
            tarch::la::Vector<DIMENSIONS, double> coords = mapMeshToCoordinates(meshPos(0), meshPos(1));

            double X = meshPos(0);
            double Y = meshPos(1);

            //double bathymetry = bathymetryHelper.getHeight(coords(0),coords(1));
            double bathymetry = dem(coords(0), coords(1));
            double mapvalue = mapMeshToMap(meshPos);


            //std::cout << "x " << X << " y " << Y << " c0 " << coords(0) << " c1 " << coords(1) << " b " << bathymetry << std::endl; 

            //std::cout << "bathymetry: " << bathymetry << " @ " << coords(0) << " " << coords(1) << std::endl;

            double r = sqrt((X-x0)*(X-x0) + (Y-y0)*(Y-y0));
            double h = hl*(r<=radDam) + hr*(r>radDam);
            double u = hl*ul*(r<=radDam) + hr*ur*(r>radDam);
            double v = hl*vl*(r<=radDam) + hr*vr*(r>radDam);

#if defined(BREAKINGDAMTEST)
            bathymetry = 0.0;
#endif

            patch.setValueUNew(subcellIndex, 0, h);
            patch.setValueUNew(subcellIndex, 1, u);
            patch.setValueUNew(subcellIndex, 2, v);
            patch.setValueUNew(subcellIndex, 3, bathymetry);
            patch.setValueAux(subcellIndex, 0, mapvalue);
 
            patch.setValueUNew(subcellIndex, 4, h * u);
            patch.setValueUNew(subcellIndex, 5, h * v);
        }
    }

    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = patch.getSubdivisionFactor();
    double min_domainsize = std::min(x_size,y_size);
    int max_subdivisionFactor = std::max(subdivisionFactor(0),subdivisionFactor(1));
}

tarch::la::Vector<DIMENSIONS,double> MekkaFlood_SWEKernelScenario::computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing) {
    double retval = 0.0;

    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();

    tarch::la::Vector<DIMENSIONS, double> patchCenter;
    patchCenter(0) = patchPosition(0) + patchSize(0)/2.0f;
    patchCenter(1) = patchPosition(1) + patchSize(1)/2.0f;

    double outerRadius = std::max(patchSize(0),patchSize(1))/2.0*sqrt(2);

    //const tarch::la::Vector<DIMENSIONS, double> mekkaPosition = mapCoordinatesToMesh(mekka_lon,mekka_lat);

    // check if mekka is inside or at least near to our current patch
    //double mekka_distance = sqrt((mekkaPosition(0)-patchCenter(0))*(mekkaPosition(0)-patchCenter(0)) + (mekkaPosition(1)-patchCenter(1))*(mekkaPosition(1)-patchCenter(1)));
 
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = patch.getSubdivisionFactor();
  
    // SCALE HERE
    double x_size = (dem.upper_right(0) - dem.lower_left(0)) / scale;
    double y_size = (dem.upper_right(1) - dem.lower_left(1)) / scale;
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;

#if 0
    double x0=(x_size) * 0.5;
    double y0=(y_size) * 0.5;
    double radDam = 0.05*std::min(x_size,y_size);
    bool isInsideCircle = false;
    for (int yi = -1; yi < subdivisionFactor(1)+1; yi++) {
        for (int xi = -1; xi < subdivisionFactor(0)+1; xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
 
            double X = patchPosition(0) + xi*meshWidth(0);
            double Y = patchPosition(1) + yi*meshWidth(1);
  
            tarch::la::Vector<DIMENSIONS, double> coords = mapMeshToCoordinates(X, Y);
            //double bathymetry = bathymetryHelper.getHeight(coords(0),coords(1));
            //double bathymetry = dem(0.0, 0.0);

            double r = sqrt((X-x0)*(X-x0) + (Y-y0)*(Y-y0));

#if defined(BREAKINGDAMTEST)
            bathymetry = 0.0;
#endif
            
            if (yi >= 0 && xi >= 0 
                && yi < subdivisionFactor(1) && xi < subdivisionFactor(0)
               ) {
                //patch.setValueAux(subcellIndex, 0,  bathymetry);
            }
            //patch.setValueUOld(subcellIndex, 3, bathymetry);

            isInsideCircle |=(r <radDam);
        }
    }
    
    // try to adapt to bathymetry
    // loop starts at one, due to central finite differences
    double max_curvature = 0.0; // second derivative
    for (int yi = 1; yi < subdivisionFactor(1)-1; yi++) {
        for (int xi = 1; xi < subdivisionFactor(0)-1; xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
            double bathemetry_11= patch.getValueAux(subcellIndex, 0);

            subcellIndex(0) = xi-1;
            subcellIndex(1) = yi;
            double bathemetry_01= patch.getValueAux(subcellIndex, 0);
 
            subcellIndex(0) = xi+1;
            subcellIndex(1) = yi;
            double bathemetry_21= patch.getValueAux(subcellIndex, 0);
 
            double curvature_x = std::abs((bathemetry_21 + 2*bathemetry_11 + bathemetry_01)/meshWidth(0));

            subcellIndex(0) = xi;
            subcellIndex(1) = yi-1;
            double bathemetry_10= patch.getValueAux(subcellIndex, 0);
 
            subcellIndex(0) = xi;
            subcellIndex(1) = yi+1;
            double bathemetry_12= patch.getValueAux(subcellIndex, 0);

            double curvature_y = std::abs((bathemetry_12 + 2*bathemetry_11 + bathemetry_10)/meshWidth(1));

            max_curvature = std::max(max_curvature, std::max(curvature_x,curvature_y));
        }
    }


      // compute spatial gradient for u and v
      double dx = patch.getSubcellSize()(0);
      double dy = patch.getSubcellSize()(1);
      
      double max_gradient = 0.0;
      double max_q0 = 0.0;
      double min_q0 = 0.0;
 
      subcellIndex(0) = 0;
      subcellIndex(1) = 0;
      max_q0 = patch.getValueUOld(subcellIndex, 0);
      min_q0 = patch.getValueUOld(subcellIndex, 0);

      // ensure that gradients between patches are smooth
 
      for (int y=0; y < subdivisionFactor(1); y++) {
          // left boundary
          {
            int x = 0;
            subcellIndex(0) = x-1;
            subcellIndex(1) = y;
            double u_01 = patch.getValueUOld(subcellIndex, 4);
                     
            subcellIndex(0) = x+1;
            subcellIndex(1) = y;
            double u_21 = patch.getValueUOld(subcellIndex, 4);
          
            subcellIndex(0) = x;
            subcellIndex(1) = y-1;
            double v_10 = patch.getValueUOld(subcellIndex, 5);
                     
            subcellIndex(0) = x;
            subcellIndex(1) = y+1;
            double v_12 = patch.getValueUOld(subcellIndex, 5);

            double du = (u_21 - u_01)/(2*dx);
            double dv = (v_12 - v_10)/(2*dy);

            max_gradient = std::max(max_gradient, std::abs(du));
            max_gradient = std::max(max_gradient, std::abs(dv));
        }

          // right boundary
          {
            int x = subdivisionFactor(0)-1;
            subcellIndex(0) = x-1;
            subcellIndex(1) = y;
            double u_01 = patch.getValueUOld(subcellIndex, 4);
                     
            subcellIndex(0) = x+1;
            subcellIndex(1) = y;
            double u_21 = patch.getValueUOld(subcellIndex, 4);
          
            subcellIndex(0) = x;
            subcellIndex(1) = y-1;
            double v_10 = patch.getValueUOld(subcellIndex, 5);
                     
            subcellIndex(0) = x;
            subcellIndex(1) = y+1;
            double v_12 = patch.getValueUOld(subcellIndex, 5);

            double du = (u_21 - u_01)/(2*dx);
            double dv = (v_12 - v_10)/(2*dy);

            max_gradient = std::max(max_gradient, std::abs(du));
            max_gradient = std::max(max_gradient, std::abs(dv));
        }

      }

      for (int x=0; x< subdivisionFactor(0); x++) {
          // bottom boundary
          {
            int y = 0;
            subcellIndex(0) = x-1;
            subcellIndex(1) = y;
            double u_01 = patch.getValueUOld(subcellIndex, 4);
                     
            subcellIndex(0) = x+1;
            subcellIndex(1) = y;
            double u_21 = patch.getValueUOld(subcellIndex, 4);
          
            subcellIndex(0) = x;
            subcellIndex(1) = y-1;
            double v_10 = patch.getValueUOld(subcellIndex, 5);
                     
            subcellIndex(0) = x;
            subcellIndex(1) = y+1;
            double v_12 = patch.getValueUOld(subcellIndex, 5);

            double du = (u_21 - u_01)/(2*dx);
            double dv = (v_12 - v_10)/(2*dy);

            max_gradient = std::max(max_gradient, std::abs(du));
            max_gradient = std::max(max_gradient, std::abs(dv));
        }

          // top boundary
          {
            int y = subdivisionFactor(1)-1;
            subcellIndex(0) = x-1;
            subcellIndex(1) = y;
            double u_01 = patch.getValueUOld(subcellIndex, 4);
                     
            subcellIndex(0) = x+1;
            subcellIndex(1) = y;
            double u_21 = patch.getValueUOld(subcellIndex, 4);
          
            subcellIndex(0) = x;
            subcellIndex(1) = y-1;
            double v_10 = patch.getValueUOld(subcellIndex, 5);
                     
            subcellIndex(0) = x;
            subcellIndex(1) = y+1;
            double v_12 = patch.getValueUOld(subcellIndex, 5);

            double du = (u_21 - u_01)/(2*dx);
            double dv = (v_12 - v_10)/(2*dy);

            max_gradient = std::max(max_gradient, std::abs(du));
            max_gradient = std::max(max_gradient, std::abs(dv));
        }

      }

#endif
 
    double max_meshwidth = std::max(meshWidth(0),meshWidth(1));
    double min_meshwidth = std::min(meshWidth(0),meshWidth(1));

#if 0
    // plain value based
    double interpolation_error_height = interpolation_error(patch, 0);
    double interpolation_error_u = interpolation_error(patch, 1);
    double interpolation_error_v = interpolation_error(patch, 2);
    //double interpolation_error_z = interpolation_error(patch, 3);

    double max_interpolation_error = 0.0;
    //max_interpolation_error = std::max(max_interpolation_error, interpolation_error_height);
    //max_interpolation_error = std::max(max_interpolation_error, interpolation_error_u);
    //max_interpolation_error = std::max(max_interpolation_error, interpolation_error_v);
    //max_interpolation_error = std::max(max_interpolation_error, interpolation_error_z);
  
    // experiment: scale error with meshwidth ( not to bad )
    max_interpolation_error = max_interpolation_error / min_meshwidth;
#endif

#if 1
    // gradient based
    double interpolation_error_gradient_height = interpolation_error_gradient(patch, 0);
    double interpolation_error_gradient_u = interpolation_error_gradient(patch, 1);
    double interpolation_error_gradient_v = interpolation_error_gradient(patch, 2);
    //double interpolation_error_gradient_z = interpolation_error_gradient(patch, 3);

    double max_interpolation_error_gradient = 0.0;
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_height);
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_u);
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_v);
    //max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_z);
#endif

#if 0
    // gradient based between coarse and fine grid
    double interpolation_error_gradient_height = interpolation_error_coarse_fine_gradient(patch, 0);
    double interpolation_error_gradient_u = interpolation_error_coarse_fine_gradient(patch, 1);
    double interpolation_error_gradient_v = interpolation_error_coarse_fine_gradient(patch, 2);
    //double interpolation_error_gradient_z = interpolation_error_coarse_fine_gradient(patch, 3);

    double max_interpolation_error_gradient = 0.0;
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_height);
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_u);
    max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_v);
    //max_interpolation_error_gradient = std::max(max_interpolation_error_gradient, interpolation_error_gradient_z);
#endif

    double min_domainsize = std::min(x_size,y_size);
    int min_subdivisionFactor = std::min(subdivisionFactor(0),subdivisionFactor(1));
    int max_subdivisionFactor = std::max(subdivisionFactor(0),subdivisionFactor(1));
 
    // AMR requires some time to rest, so lets give the patches a little bit more time
    //if (patch.getAge() >= 2) { // 2 works in sequential mode
        retval = max_meshwidth;
        //std::cout << "interpolation error inside patch: " << max_error << " " << " with ghostlayer " << max_error_ghost << std::endl;

        //std::cout << "max interpolation error " << max_interpolation_error << std::endl;
        //std::cout << "max interpolation error gradient " << max_interpolation_error_gradient << std::endl;
        if (max_interpolation_error_gradient > 1.e-1) { // 1.e-2 and basic interpolation gradient used for POSTER
            //std::cout << "interpolation error inside patch: " << max_error << " " << " with ghostlayer " << max_error_ghost << std::endl;
            //std::cout << "refining!" << std::endl;
//            retval = retval / (3.0 * max_subdivisionFactor);
            retval = x_size/patch.getSubdivisionFactor()(0)/81.0;
        } else {
//            retval = retval * (3.0 * max_subdivisionFactor);
            retval = x_size/patch.getSubdivisionFactor()(0)/9.0;
        }
        patch.resetAge();
//    } else {
//        retval = max_meshwidth;
//    }
  
    // ensure minimum refinement
    if (retval > (min_domainsize / (3.0 * max_subdivisionFactor))) {
        //std::cout << "refining  " << retval << " vs " << min_domainsize / (1.0 * max_subdivisionFactor) <<  std::endl;
        retval = min_domainsize / (3.0 * max_subdivisionFactor);
        patch.resetAge();
    }
 
    // ensure maximum refinement
    if (retval < (min_domainsize / (9.0 * max_subdivisionFactor))) { 
        //std::cout << "too small refinement!!! " << retval << " vs " << (min_domainsize / (81.0 * max_subdivisionFactor)) <<  std::endl;
        retval = min_domainsize / (9.0 * max_subdivisionFactor);
        patch.resetAge();
    }

    /*if (mekka_distance > outerRadius) {
        retval = domainSize/6/243;
    } else {
        //retval = 1e-3;
        retval = domainSize/6/243;
    }*/

    /*if (meshWidth(0) < 1e-4 || max_curvature > 1e1) {
        retval = meshWidth(0);
    } else {
        //std::cout << "max_curvature is: " << max_curvature << std::endl;
        retval = meshWidth(0)/3.0;
    }*/

    /*if (isInsideCircle && meshWidth(0) > 3e-3) {
        retval = meshWidth(0)/3.0;
    } else {
        retval = meshWidth(0);
    }*/
    return retval;

    //TODO unterweg debug
//    if(std::abs(patch.getPosition()(0) - 3000) < 500 && patch.getLevel() < 5 && !isInitializing) {
//      return patch.getSubcellSize() / 3.0;
////      return x_size/16/27.0;
//    } else if( std::abs(patch.getPosition()(0) - 3000) < 500 ){
//      return patch.getSubcellSize();
//    } else {
//      return patch.getSubcellSize() * 3.0;
////      return x_size/16/9.0;
//    }
}
 
void MekkaFlood_SWEKernelScenario::update(peanoclaw::Patch& patch) {
    // update bathymetry data
    //std::cout << "updating bathymetry!" << std::endl;
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    tarch::la::Vector<DIMENSIONS, double> meshPos;
    for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
        for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
            subcellIndex(0) = xi;
            subcellIndex(1) = yi;
 
            meshPos = patch.getSubcellPosition(subcellIndex);
            tarch::la::Vector<DIMENSIONS, double> coords = mapMeshToCoordinates(meshPos(0), meshPos(1));
            double bathymetry = dem(coords(0), coords(1));
            double mapvalue = mapMeshToMap(meshPos);

            patch.setValueUNew(subcellIndex, 3, bathymetry);
            patch.setValueAux(subcellIndex, 0, mapvalue);
        }
    }
}

// box sizes:
// - complete saudi arabia:
//  longitude: 32 - 60E
//  latitude: 13 - 40N
//
// - mekka area


tarch::la::Vector<DIMENSIONS, double> MekkaFlood_SWEKernelScenario::mapCoordinatesToMesh(double longitude, double latitude) {
    tarch::la::Vector<DIMENSIONS, double> mesh;
 
    /*// put mekkah in the center - adjust bei 0*5 * scale / domainSize

    double scale = 0.1;

    // peano coordinates
    //float mesh_y = (latitude-13.0f)/27.0f * domainSize;
    //float mesh_x = (longitude-32.0f)/28.0f * domainSize;
 
    float mesh_y = (latitude-39.8167f)/1.0f * domainSize / scale + 0.5;
    float mesh_x = (longitude-21.4167f)/1.0f * domainSize / scale + 0.5;

    mesh(0) = mesh_x;
    mesh(1) = mesh_y;*/
	

    double lower_left_0 = dem.lower_left(0);
    double lower_left_1 = dem.lower_left(1);
 
    std::cout << "lower left " << lower_left_0 << " " << lower_left_1 << std::endl;

    double ws_0;
    double ws_1;
    dem.pixelspace_to_worldspace(ws_0, ws_1, longitude, latitude);

    // TODO_: the offset change is just due to a problem with plotting, meh ...
    mesh(0) = ws_0 - lower_left_0;
    mesh(1) = ws_1 - lower_left_1;
    return mesh;
}

tarch::la::Vector<DIMENSIONS, double> MekkaFlood_SWEKernelScenario::mapMeshToCoordinates(double x, double y) {
    tarch::la::Vector<DIMENSIONS, double> coords;

    // put mekkah in the center - adjust bei 0*5 * scale / domainSize

    /*double scale = 0.5;

    //double latitude = x / domainSize * 27.0 + 13.0;
    //double longitude = y / domainSize * 28.0 + 32.0;
 
    double latitude = (y-0.5)* (scale / domainSize) * 1.0 + 21.4167;
    double longitude = (x-0.5)* (scale / domainSize) * 1.0 + 39.8167;
    coords(0) = longitude;
    coords(1) = latitude;*/
 
    double lower_left_0 = dem.lower_left(0);
    double lower_left_1 = dem.lower_left(1);
 
    double ps_0 = 0.0;
    double ps_1 = 0.0;
    dem.worldspace_to_pixelspace(ps_0, ps_1, x*scale+lower_left_0, y*scale+lower_left_1);


    coords(0) = ps_0;
    coords(1) = ps_1;
    return coords;
}

double MekkaFlood_SWEKernelScenario::mapMeshToMap(tarch::la::Vector<DIMENSIONS, double>& coords) {
#if 0
    // relate pixel in png file to bathymetry data
    int width_map = mekka_map.width;
    int height_map = mekka_map.height;
    double width_domain = (dem.upper_right(0) - dem.lower_left(0));
    double height_domain = (dem.upper_right(1) - dem.lower_left(1));

    // relate desired coords to data in map
    double map_pos_x = coords(0)*scale * (width_map / width_domain);
    double map_pos_y = height_map - (coords(1)*scale * (height_map / height_domain)); // picture is upside down otherwise :D

    // now map position to index in png file
    int map_index_x = std::floor(map_pos_x);
    int map_index_y = std::floor(map_pos_y);

    // buffer is organized as grayscale one byte per pixel
    int bufferpos = map_index_y * width_map * 1 + map_index_x * 1;

    // convert gray scale to a double precision value
    double gray_value = mekka_map_data[bufferpos] / 255.0;
#endif
    double gray_value = 0.0;
    return gray_value;
}

#endif
