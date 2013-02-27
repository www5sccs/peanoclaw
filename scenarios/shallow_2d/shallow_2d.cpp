#include <iostream>
#include <ostream>
#include <fstream>
#include <cmath>
#include <cstring>

void qinit(double xlower, double ylower,
           double xupper, double yupper,
           double hl, double ul, double vl, 
           double hr, double ur, double vr,
           double radDam,
           //----------
           const int num_equations, // must be 3 for this particular example
           const int num_cells_x,
           const int num_cells_y,
           const int num_ghost,
           const int ldq_y,
           double *q
           )
{
    double x0 = 0.5;
    double y0 = 0.5;

    double cell_size_x = (xupper - xlower) / num_cells_x;
    double cell_size_y = (yupper - ylower) / num_cells_y;
 
    int x_stride = num_equations;
    int ghost_skip = num_ghost * ldq_y + num_ghost * x_stride;

    for (int cells_x=-num_ghost; cells_x < num_cells_x + num_ghost; cells_x++) {
        double xCenter = xlower + cells_x * cell_size_x + cell_size_x * 0.5;
        for (int cells_y=-num_ghost; cells_y < num_cells_y + num_ghost; cells_y++) {
            double yCenter = ylower + cells_y * cell_size_y + cell_size_y * 0.5;
            double r = sqrt((xCenter-x0)*(xCenter-x0) + (yCenter-y0)*(yCenter-y0));

            // don't forget, we need FORTRAN style: column-major order
            q[cells_y*ldq_y+cells_x*x_stride+ghost_skip+0] = hl*(r<=radDam) + hr*(r>radDam);
            q[cells_y*ldq_y+cells_x*x_stride+ghost_skip+1] = hl*ul*(r<=radDam) + hr*ur*(r>radDam);
            q[cells_y*ldq_y+cells_x*x_stride+ghost_skip+2] = hl*vl*(r<=radDam) + hr*vr*(r>radDam);
        }
    }
}

// TODO: why is maxm an argument? in AMRCLAW it's defined as maxm = max(mx,my)

typedef void (*rpn2_f)(int *ixy_p, int *maxm_p, int *meqn_p, int *mwaves_p, int *mbc_p, int *mx_p,
                       double *ql, double *qr, double *auxl, double *auxr, double *wave, double *s,  
                       double *amdq, double *apdq, int *num_aux_p
                      );

typedef void (*rpt2_f)(int *ixy_p, int *maxm_p, int *meqn_p, int *mwaves_p, int *mbc_p, int *mx_p,
                       double *ql, double *qr, double *aux1, double *aux2, double *aux3, int *ilr_p, 
                       double *asdq, double *bmasdq, double *bpasdq, int *num_aux_p
                      );

/*extern void step2_kaust(int *maxm_p, int *num_eqn_p, int *num_waves_p, int *num_aux_p, int *num_ghost_p, 
                  int *mx_p, int *my_p, double *q_old, double *q_new, double *aux, 
                  double *dx_p, double *dy_p, double *dt_p, int *methodarray, int *mthlimarray, double *cfl_p,
                  double *qadd, double *fadd, double *gadd, double *aux1, double *aux2, double *aux3,
                  double *work, int *worksize_p, bool use_fwave, 
                  (*rpn2)
                  (*rpt2)
                  );*/

extern "C" void rpn2_shallow_roe_with_efix_(int *ixy_p, int *maxm_p, int *meqn_p, int *mwaves_p, int *mbc_p, int *mx_p,
                       double *ql, double *qr, double *auxl, double *auxr, double *wave, double *s,  
                       double *amdq, double *apdq, int *num_aux_p
                      );

extern "C" void rpt2_shallow_roe_with_efix_(int *ixy_p, int *maxm_p, int *meqn_p, int *mwaves_p, int *mbc_p, int *mx_p,
                       double *ql, double *qr, double *aux1, double *aux2, double *aux3, int *ilr_p, 
                       double *asdq, double *bmasdq, double *bpasdq, int *num_aux_p
                      );

extern "C" void step2_(int *maxm_p, int *maxmx_p, int *maxmy_p, int *num_eqn_p, int *num_waves_p, int *num_aux_p, int *num_ghost_p,
                  int *mx_p, int *my_p, int *mcapa_p, int *method, double *mthlim,
                  double *q_old, double *aux,
                  double *dx_p, double *dy_p, double *dt_p, double *cflgrid, 
                  double *fm, double *fp, double *gm, double *gp,
                  rpn2_f rpn2,
                  rpt2_f rpt2
                 );

void writeq(std::ostream& stream,
            const char *title,
            const int selected_equation,
            const int num_equations,
            const int num_cells_x,
            const int num_cells_y,
            const int num_ghost,
            const int ldq_y,
            double *q
) {
    //stream << "# " << title << " " << num_equations << " " << num_cells_x << " " << num_cells_y << " " << num_ghost << std::endl;
    int x_stride = num_equations;
    int ghost_skip = num_ghost * ldq_y + num_ghost * x_stride;
    for (int cells_x = -num_ghost; cells_x < num_cells_x+num_ghost; cells_x++) {
        if (selected_equation == -1) {
            for (int var = 0; var < num_equations; var++) {
                for (int cells_y = -num_ghost; cells_y < num_cells_x+num_ghost; cells_y++) {
                    int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
                    stream << q[index+var] << " ";
                }
                stream << std::endl;
            }
            stream << "-----" << std::endl;
        } else {
            int var = selected_equation;
            for (int cells_y = -num_ghost; cells_y < num_cells_x+num_ghost; cells_y++) {
                int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
                stream << cells_x << " " << cells_y <<  " " << q[index+var] << std::endl;
            }
        }
    }
}

void boundary_conditition_wall( const int num_equations,
                                const int num_cells_x,
                                const int num_cells_y,
                                const int num_ghost,
                                const int ldq_y,
                                double *q
                              )
{
    int x_stride = num_equations;
    int ghost_skip = num_ghost * ldq_y + num_ghost * x_stride;


    // handle ghost cells on the left
    for (int cells_y = -num_ghost; cells_y < 0; cells_y++) {
        for (int cells_x = -num_ghost; cells_x < num_cells_x+num_ghost; cells_x++) {
            int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
            q[index+1] = -q[index+1];
        }
    }

    // handle ghost cells on the right
    for (int cells_y = num_cells_y; cells_y < num_cells_y+num_ghost; cells_y++) {
        for (int cells_x = -num_ghost; cells_x < num_cells_x+num_ghost; cells_x++) {
            int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
            q[index+1] = -q[index+1];
        }
    }

    for (int cells_y = -num_ghost; cells_y < num_cells_y+num_ghost; cells_y++) {
        // handle upper ghost cells
        for (int cells_x = -num_ghost; cells_x < 0; cells_x++) {
            int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
            q[index+1] = -q[index+1];
        }
    
        // handle lower ghost cells
        for (int cells_x = num_cells_x; cells_x < num_cells_x+num_ghost; cells_x++) {
            int index = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
            q[index+1] = -q[index+1];
        }
    }
}

int main(int argc, char **argv) {
    // Domain
    double xlower = 0;
    double ylower = 0;

    double xupper = 1;
    double yupper = 1;

    int num_equations = 3; // TODO: actually determined by chosen riemann solver
    int num_waves = 3; // TODO: actually determined by chosen riemann solver
    int num_cells_x = 40;
    int num_cells_y = 40;

    // Riemann States of the dam break problem
    const double damRadius = 0.5;
    const double hl = 5;
    const double ul = 0;
    const double vl = 0;
    const double hr = 1;
    const double ur = 0;
    const double vr = 0;

    // solver properties
    double dt_initial = 0.005;
    int num_aux = 1; // pyclaw works around a bug in f2py and sets this to 1 (zero size arrays)
    int num_ghost = 1;

    // allocate q (without ghost cells)
    int maxmx = num_cells_x;
    int maxmy = num_cells_y;
    int maxm = std::max(maxmx,maxmy); // largest stride
    int ldq_y = (maxmx+2*num_ghost)*num_equations;
    int q_elements = ldq_y*(maxmy+2*num_ghost);
    double *q = new double[q_elements];
    double *q_old = new double[q_elements];

    // allocate aux (without gost cells)
    int aux_elements = ldq_y*(maxmy+2*num_ghost)*num_aux;
    double *aux = new double[aux_elements];

    // allocate fm,fp,gm,gp (without ghost cells)
    double *fm = new double[q_elements];
    double *fp = new double[q_elements];
    double *gm = new double[q_elements];
    double *gp = new double[q_elements];

    // seed initial solution
    memset(q_old, 0, q_elements*sizeof(double));
    memset(q, 0, q_elements*sizeof(double));
    memset(aux, 0, aux_elements*sizeof(double));

    qinit(xlower,ylower,
          xupper,yupper,
          hl,ul,vl,
          hr,ur,vr,
          damRadius,
          num_equations,
          num_cells_x,
          num_cells_y,
          num_ghost,
          ldq_y,
          q_old
          );

    memcpy(q, q_old, q_elements*sizeof(double));

    // engage riemann solver
    double dx = (xupper - xlower) / num_cells_x; // also used in qinit
    double dy = (yupper - ylower) / num_cells_y;
    double dt = dt_initial;
    double cflgrid = 0.0; // return value
    int mcapa = 0;
    int method[7];
    method[0] = 0;
    method[1] = 1;
    method[2] = 0; 
    method[3] = 0;
    method[4] = 0;
    method[5] = 0;
    method[6] = 0;

    double *mthlim = new double[num_waves];
    mthlim[0] = 0.0;
    mthlim[1] = 0.0;
    mthlim[2] = 0.0;

    char filename[256];

    for (int i = 0; i < 1000; i++) {

        // apply boundary conditions 
        boundary_conditition_wall(num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,q);

        // TODO: apply auxillary conditions/values/parameters?

        // perform flux computation
        step2_(&maxm, &maxmx, &maxmy, &num_equations, &num_waves, &num_aux, &num_ghost,
              &num_cells_x, &num_cells_y, &mcapa, method, mthlim,
              q, aux,
              &dx, &dy, &dt, &cflgrid, 
              fm, fp, gm, gp,
              rpn2_shallow_roe_with_efix_,
              rpt2_shallow_roe_with_efix_
             );
        
        sprintf(filename, "shallow_2d-%i.dat", i);

        std::ofstream plotoutput(filename);
        std::cout << "# cfl " << cflgrid << " dx " << dx << " dy " << dy << " dt " << dt << std::endl;
        writeq(plotoutput, "q0", 0, num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,q);

        //printq("fm", num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,fm);
        //printq("fp", num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,fp);
        //printq("gm", num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,gm);
        //printq("gp", num_equations,num_cells_x,num_cells_y,num_ghost,ldq_y,gp);

        // update q (analog to AMRClaw stepgrid): apply fluxes
        int x_stride = num_equations;
        int ghost_skip = num_ghost * ldq_y + num_ghost * x_stride;

        double dtdx = dt / dx;
        double dtdy = dt / dy;

        for (int cells_y=-num_ghost; cells_y < num_cells_y + num_ghost; cells_y++) {
            for (int cells_x=-num_ghost; cells_x < num_cells_x + num_ghost; cells_x++) {
                // don't forget, we need FORTRAN style: column-major order
                int index1 = cells_y*ldq_y+cells_x*x_stride+ghost_skip;
                int index2 = cells_y*ldq_y+(cells_x+1)*x_stride+ghost_skip;
                int index3 = (cells_y+1)*ldq_y+cells_x*x_stride+ghost_skip;
                for (int var=0; var < num_equations; var++) {
                    q[index1+var] = q[index1+var] - dtdx * (fm[index2+var] - fp[index1+var])
                                              - dtdy * (gm[index3+var] - gp[index1+var]);
                }
            }
        }

        plotoutput.close();
    }

    // cleanup
    delete[] mthlim;
    delete[] fm;
    delete[] fp;
    delete[] gm;
    delete[] gp;
    delete[] aux;
    delete[] q;
    delete[] q_old;
    return 0;
}
