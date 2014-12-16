//// BASED ON FULLSWOF2D SOLVER
//
//#ifndef _MEKKAFLOOD_SOLVER_H_
//#define _MEKKAFLOOD_SOLVER_H_
//
//#include <limits>
//
//class MekkaFlood_solver {
//    protected:
//
//    private:
//        // this structs are not allocated and serve just as links to temp data
//        struct SchemeArrays {
//            double *h;
//            double *u;
//            double *v;
//            double *q1;
//            double *q2;
//            double *z;
//        };
//
//    public:
//        struct InputArrays {
//             // 0 = stride between elements, 1 stride between rows, 2 stride between patches
//
//            double *h;
//            double *u;
//            double *v;
//            double *z;
//        };
//
//        struct TempArrays {
//            // set by MUSCL reconstruction
//            double *h1r;
//            double *h1l;
//
//            double *u1r;
//            double *u1l;
//
//            double *v1r;
//            double *v1l;
//
//            double *z1r;
//            double *z1l;
//            double *delta_z1; // maybe we can rules this one away its just: z[i+1][j]-z[i][j];
//            double *delzc1;
//            double *delz1;
//
//            double *h2r;
//            double *h2l;
//
//            double *u2r;
//            double *u2l;
//
//            double *v2r;
//            double *v2l;
//
//            double *z2r;
//            double *z2l;
//            double *delta_z2; // maybe we can rules this one away its just: z[i+1][j]-z[i][j];
//            double *delzc2;
//            double *delz2;
//
//
//            // flux computation
//            double *f1;
//            double *f2;
//            double *f3;
//
//            double *g1;
//            double *g2;
//            double *g3;
//
//            // scheme
//            double *q1;
//            double *q2;
//
//            double *hs;
//            double *us;
//            double *vs;
//            double *qs1;
//            double *qs2;
//
//            double *hsa;
//            double *usa;
//            double *vsa;
//            double *qsa1;
//            double *qsa2;
//
//            double *Vin1;
//            double *Vin2;
//            double *Vin_tot;
//
//            // friction
//            double *Fric_tab;
//
//            // set by hydrostatic reconstruction
//            double *h1right;
//            double *h1left;
//            double *h2right;
//            double *h2left;
//
//            // rain:
//            double *Tab_rain;
//        };
//
//
//        struct Constants {
//            Constants(int nx, int ny, double dx, double dy) :
//                NXCELL(nx),
//                NYCELL(ny),
//                DX(dx),
//                DY(dy)
//            {
//                GRAVITATION = 9.81;
//                GRAVITATION_DEM = 4.905;
//                CONST_CFL_X = 0.5;
//                CONST_CFL_Y = 0.5;
//                HE_CA = 1.e-12;
//                VE_CA = 1.e-12;
//                MAX_CFL_X = 0.;
//                MAX_CFL_Y = 0.;
//                NB_CHAR = 256;
//                ZERO = 0.;
//                IE_CA = 1.e-8;
//                EPSILON = 1.e-13;
//                Ratio_Close_cell = 1e-3;
//                MAX_SCAL = std::numeric_limits<double>::max();
//                RainIntensity = 0.00001;
//
//                FRICCOEF = 0.0; // TODO get real value for this
//                CFL_FIX = 0.5;
//            }
//
//            double GRAVITATION; // 9.81
//
//            double GRAVITATION_DEM; // 4.905
//            double CONST_CFL_X; // 0.5
//            double CONST_CFL_Y; // 0.5
//            double HE_CA; //1e-12;
//            double VE_CA; //1e-12;
//
//            double MAX_CFL_X; // 0.
//            double MAX_CFL_Y; // 0.
//
//            int NB_CHAR; // 256 // TODO: what is its use?
//            double ZERO; // 0
//            double IE_CA; // 1.e-8;
//            double EPSILON; // 1.e-13
//
//            double Ratio_Close_cell; // 1e-3
//            double MAX_SCAL; // DBL_MAX
//
//            double RainIntensity; // 0.001;
//
//            double FRICCOEF;
//            double CFL_FIX;
//
//            const int NXCELL;
//            const int NYCELL;
//            const double DX;
//            const double DY;
//        };
//
//        // index helper
//        static inline unsigned int linearizeIndex(int dim, unsigned int* index, unsigned int* strides) {
//            unsigned int result = 0;
//            for (int i=0; i < dim; i++) {
//                result += index[i] * strides[i];
//            }
//            return result;
//        }
//
//        static void initializeStrideinfo(const Constants& constants, int dim, unsigned int* strideinfo);
//        static void allocateInput(int nr_patches, int dim, unsigned int* strideinfo, InputArrays& input);
//        static void allocateTemp(int nr_patches, int dim, unsigned int* strideinfo, TempArrays& temp);
//        static void freeInput(InputArrays& input);
//        static void freeTemp(TempArrays& temp);
//
//        MekkaFlood_solver();
//        virtual ~MekkaFlood_solver();
//
//        // returns used timestep
//        static double calcul(const int patchid, int dim, unsigned int* strideinfo, InputArrays& input, TempArrays& temp, const Constants& constants, double dt_max);
//
//        static void boundary(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants,
//                             double time_tmp);
//
//        // minmod slope limiter
//        static double lim_minmod(double a, double b);
//
//        // muscl reconstruction
//        static void rec_muscl_init(const int patchid, int dim, unsigned int* strideinfo, InputArrays& input, TempArrays& temp, const Constants& constants);
//        static void rec_muscl(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants);
//
//        // hydrostatic reconstruction
//        static void rec_hydro(double hg, double hd,double dz, double& hg_rec, double& hd_rec);
//
//        // HLL flux
//        static void flux_hll(const Constants& constants, double h_L,double u_L,double v_L,double h_R,double u_R,double v_R, double& f1, double& f2, double& f3, double& cfl);
//
//        // rain
//        static void rain(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants,
//                         double time);
//
//        // friction: (actually no friction)
//        // CAREFUL: input.q1 == output.q1 and input.q2 == output.q2
//        static void friction(double uold, double vold, double hnew, double q1new, double q2new, double dt, double cf, double& q1mod, double& q2mod);
//
//        // infiltration: (actually no infiltration)
//        static void infiltration(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants,
//                                 double dt);
//
//        // general scheme:
//        static void maincalcflux(const int patchid, int dim, unsigned int* strideinfo, TempArrays& temp, const Constants& constants,
//                                 double cflfix, double dt_max, double& dt); // dt is both input AND output
//        static void maincalcscheme(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, SchemeArrays& output, TempArrays& temp, const Constants& constants,
//                                   double tps, double dt, int verif); // there was originally a "n" input parameter which was not used here.
//};
//
//#endif // _MEKKAFLOOD_SOLVER_H_
