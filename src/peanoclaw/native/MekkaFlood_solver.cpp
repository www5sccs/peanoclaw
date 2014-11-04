// BASED ON FULLSWOF2D Solver

#include <sys/time.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "peanoclaw/native/MekkaFlood_solver.h"

// a fused version of the original components
// - scheme: order2
// - reconstruction: muscl
// - slope limiter: minmod
// - flux: HLL
// - infiltration: none
// - friction: none

const int VECTOR_LENGTH = 4;

static const double FZERO = 0;

MekkaFlood_solver::MekkaFlood_solver()
{

}


MekkaFlood_solver::~MekkaFlood_solver() {

}
 
void MekkaFlood_solver::initializeStrideinfo(const Constants& constants, int dim, unsigned int* strideinfo) {
    int nx_ghost = constants.NXCELL + 2;
    int ny_ghost = constants.NYCELL + 2;

    strideinfo[0] = 1;
    strideinfo[1] = static_cast<int>(std::ceil(static_cast<double>(ny_ghost) / static_cast<double>(VECTOR_LENGTH)) * VECTOR_LENGTH); // align in to a multiple of 4 (double) elements
    strideinfo[2] = strideinfo[1] * nx_ghost;

    //std::cout << "strideinfo " << strideinfo[0] << " " << strideinfo[1] << " " << strideinfo[2] << " nx " << nx_ghost << " ny " << ny_ghost << std::endl;
}

void MekkaFlood_solver::allocateInput(int nr_patches, int dim, unsigned int* strideinfo, InputArrays& input) {
    unsigned int patchsize = strideinfo[2] * sizeof(double);
    unsigned int totalsize = patchsize * nr_patches;

    posix_memalign(reinterpret_cast<void**>(&(input.h)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(input.u)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(input.v)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(input.z)), VECTOR_LENGTH*sizeof(double), totalsize);
}

void MekkaFlood_solver::allocateTemp(int nr_patches, int dim, unsigned int* strideinfo, TempArrays& temp) {
    unsigned int patchsize = strideinfo[2] * sizeof(double);
    unsigned int totalsize = patchsize * nr_patches;

    posix_memalign(reinterpret_cast<void**>(&(temp.h1r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.h1l)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.u1r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.u1l)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.v1r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.v1l)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.z1r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.z1l)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.delta_z1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.delzc1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.delz1)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.h2r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.h2l)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.u2r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.u2l)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.v2r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.v2l)), VECTOR_LENGTH*sizeof(double), totalsize);
  
    posix_memalign(reinterpret_cast<void**>(&(temp.z2r)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.z2l)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.delta_z2)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.delzc2)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.delz2)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.f1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.f2)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.f3)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.g1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.g2)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.g3)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.q1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.q2)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.hs)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.us)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.vs)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.qs1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.qs2)), VECTOR_LENGTH*sizeof(double), totalsize);
  
    posix_memalign(reinterpret_cast<void**>(&(temp.hsa)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.usa)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.vsa)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.qsa1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.qsa2)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.Vin1)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.Vin2)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.Vin_tot)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.Fric_tab)), VECTOR_LENGTH*sizeof(double), totalsize);
 
    posix_memalign(reinterpret_cast<void**>(&(temp.h1right)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.h1left)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.h2right)), VECTOR_LENGTH*sizeof(double), totalsize);
    posix_memalign(reinterpret_cast<void**>(&(temp.h2left)), VECTOR_LENGTH*sizeof(double), totalsize);

    posix_memalign(reinterpret_cast<void**>(&(temp.Tab_rain)), VECTOR_LENGTH*sizeof(double), totalsize);
}

void MekkaFlood_solver::freeInput(InputArrays& input) {
    free(input.h);
    free(input.u);
    free(input.v);
    free(input.z);
}

void MekkaFlood_solver::freeTemp(TempArrays& temp) {
    // set by MUSCL reconstruction
    free(temp.h1r);
    free(temp.h1l);

    free(temp.u1r);
    free(temp.u1l);

    free(temp.v1r);
    free(temp.v1l);

    free(temp.z1r);
    free(temp.z1l);
    free(temp.delta_z1); // maybe we can rules this one away its just: z[i+1][j]-z[i][j];
    free(temp.delzc1);
    free(temp.delz1);

    free(temp.h2r);
    free(temp.h2l);

    free(temp.u2r);
    free(temp.u2l);

    free(temp.v2r);
    free(temp.v2l);

    free(temp.z2r);
    free(temp.z2l);
    free(temp.delta_z2); // maybe we can rules this one away its just: z[i+1][j]-z[i][j];
    free(temp.delzc2);
    free(temp.delz2);

    // flux computation
    free(temp.f1);
    free(temp.f2);
    free(temp.f3);

    free(temp.g1);
    free(temp.g2);
    free(temp.g3);

    // scheme
    free(temp.q1);
    free(temp.q2);

    free(temp.hs);
    free(temp.us);
    free(temp.vs);
    free(temp.qs1);
    free(temp.qs2);

    free(temp.hsa);
    free(temp.usa);
    free(temp.vsa);
    free(temp.qsa1);
    free(temp.qsa2);

    free(temp.Vin1);
    free(temp.Vin2);
    free(temp.Vin_tot);

    // friction
    free(temp.Fric_tab);

    // set by hydrostatic reconstruction
    free(temp.h1right);
    free(temp.h1left);
    free(temp.h2right);
    free(temp.h2left);

    // rain:
    free(temp.Tab_rain);
}


double MekkaFlood_solver::calcul(const int patchid, int dim, unsigned int* strideinfo, InputArrays& input, TempArrays& temp, const Constants& constants, double dt_max) {
	/**
	 * @details Performs the second order numerical scheme.
	 * @note In DEBUG mode, the programme will save another file with boundary fluxes.
	 */

    const int NXCELL = constants.NXCELL;
    const int NYCELL = constants.NYCELL;

    unsigned int index[3];

  //initialization

#if 0
  for (int i=1 ; i<=NXCELL ; i++){
    for (int j=1 ; j<=NYCELL ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
	
      //h[i][j] =0.001;
      temp.q1[centerIndex] = input.u[centerIndex]*input.h[centerIndex];
      temp.q2[centerIndex] = input.v[centerIndex]*input.h[centerIndex];
    } //end for j
  } //end for i
#endif

  for (int i=0 ; i<=NXCELL+1 ; i++){
    for (int j=0 ; j<=NYCELL+1 ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
	
      temp.Fric_tab[centerIndex]=constants.FRICCOEF;
    } //end for j
  }//end for i

#if 0
  for (int i=1 ; i<=NXCELL ; i++){
    for (int j=1 ; j<=NYCELL ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
	
        temp.Vin_tot[centerIndex] = FZERO;
    } //end for j
  } //end for i
#endif
 

  rec_muscl_init(patchid, dim, strideinfo, input, temp, constants);

  double dt2 = 0.0;
  

  int verif = 1; // TODO: this might actually go to zero
  

  //time's iteration beginning 
  //while (T > tps  && n < MAX_ITER){  // we only need one iteration so no need for a loop
 
  double dt1;
  double tps = FZERO; // TODO: current simulation time. however, we do not have time changing boundaries

  int n = 0;

  while (n < 1) {

 
    SchemeArrays sinput1;
    sinput1.h = input.h;
    sinput1.u = input.u;
    sinput1.v = input.v;
    sinput1.z = input.z;
    sinput1.q1 = temp.q1;
    sinput1.q2 = temp.q2;
 
    SchemeArrays sinput2;
    sinput2.h = temp.hs;
    sinput2.u = temp.us;
    sinput2.v = temp.vs;
    sinput2.z = input.z;
    sinput2.q1 = temp.qs1;
    sinput2.q2 = temp.qs2;


    if (1 == verif){
      dt1=dt_max;
 
      //boundary conditions
      boundary(patchid, dim, strideinfo, sinput1, temp, constants, tps);
       

      for (int i=1 ; i<NXCELL+1 ; i++){
        for (int j=1 ; j<NYCELL+1 ; j++){
            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
        
          if (input.h[centerIndex]<=constants.HE_CA) {
            input.h[centerIndex]=FZERO;
            input.u[centerIndex] = FZERO;
            input.v[centerIndex] = FZERO;
          }
          if (fabs(input.u[centerIndex])<=constants.VE_CA){
            input.u[centerIndex]=FZERO;
            temp.q1[centerIndex]=FZERO;
          }
          if (fabs(input.v[centerIndex])<=constants.VE_CA){
            input.v[centerIndex]=FZERO;
            temp.q2[centerIndex]=FZERO;
          }
        } //end for j
      } //end for i
 

      // Reconstruction for order 2:
      rec_muscl(patchid, dim, strideinfo, sinput1, temp, constants);
    
    }else{//if verif==0, ie dt2<dt1
        /*We return to the value of the previous time (i.e time=n) */
      for (int i=1 ; i<NXCELL+1 ; i++) {
        for (int j=1 ; j<NYCELL+1 ; j++) {
            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
 
          temp.Vin1[centerIndex]=temp.Vin2[centerIndex];
        }
      }
    }//end for if verif==1
 
    maincalcflux(patchid, dim, strideinfo, temp, constants, constants.CFL_FIX, dt_max, dt1);
#if 0 // TODO: as this is controlled by peanoclaw we can always assume the full possible timestep
    dt1=min(T-tps,dt1);
#endif
    
    double tx=dt1/constants.DX;
    double ty=dt1/constants.DY;
  
    maincalcscheme(patchid, dim, strideinfo, sinput1, sinput2, temp, constants, tps, dt1, verif); // careful, this generate new data which will be used as input in the second step
    //    dt2=dt_max;
    dt2=dt1;    
  
  
    // set input to previously generated output for upcoming actions
    //boundary conditions
    boundary(patchid, dim, strideinfo, sinput2, temp, constants, tps+dt1);

    for (int i=1 ; i<NXCELL+1 ; i++){
      for (int j=1 ; j<NYCELL+1 ; j++){
            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            unsigned int centerIndex = linearizeIndex(3, index, strideinfo);
 
        if (temp.hs[centerIndex]<=constants.HE_CA) { // blend / select operation
          temp.hs[centerIndex]=0.;
          temp.us[centerIndex] = 0.;
          temp.vs[centerIndex] = 0.;
          temp.qs1[centerIndex]=0.;
          temp.qs2[centerIndex]=0.;
        }
        if (fabs(temp.us[centerIndex])<=constants.VE_CA){ // blend / select operation
          temp.us[centerIndex]=0.;
          temp.qs1[centerIndex]=0.;
        }
        if (fabs(temp.vs[centerIndex])<=constants.VE_CA){ // blend / select operation
          temp.vs[centerIndex]=0.;
          temp.qs2[centerIndex]=0.;
        }
      } //end for j
    } //end for i
  
    //Reconstruction for order 2 
    rec_muscl(patchid, dim, strideinfo, sinput2, temp, constants);
   
    //commun bloc
    maincalcflux(patchid, dim, strideinfo, temp, constants, constants.CFL_FIX, dt_max, dt2);
  
    std::cout << "check before verif is zero: dt2 " << dt2 << " dt 1 " << dt1 << std::endl;

    // TODO: this can not be solved by blending / select alone!
    if (dt2<dt1){ // this case actually happens in our case, when the timestep gets too large!
      dt1=dt2;
      tx=dt1/constants.DX;
      ty=dt1/constants.DY;
      verif=0;
    }else{
  
      //Added to do calculus at the beginning 
      verif=1;
      
    SchemeArrays soutput;
    soutput.h = temp.hsa;
    soutput.u = temp.usa;
    soutput.v = temp.vsa;
    soutput.z = input.z;
    soutput.q1 = temp.qsa1;
    soutput.q2 = temp.qsa2;
      
    maincalcscheme(patchid, dim, strideinfo, sinput2, soutput, temp, constants, tps, dt1, verif); // takes new input arguments  and generate and third set of input arguments
     
      /*the values of height_Vinf_tot and height_of_tot put to zero
	to compute infiltrated and overland flow volume*/
      double height_Vinf_tot=FZERO;  
      double height_of_tot=FZERO;  
      //Heun method (order 2 in time)
      for (int i=1 ; i<NXCELL+1 ; i++){
        for (int j=1 ; j<NYCELL+1 ; j++){
            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            unsigned int centerIndex = linearizeIndex(3, index, strideinfo);

          if (temp.hsa[centerIndex]<=constants.HE_CA) {  // blend / select: keep or set to zero
              temp.hsa[centerIndex]=FZERO;
          }

          if (abs(temp.usa[centerIndex])<=constants.VE_CA) { // blend / select: keep or set to zero
              temp.usa[centerIndex]=FZERO;
          }

          if (abs(temp.vsa[centerIndex])<=constants.VE_CA) { // blend / select: keep or set to zero
              temp.vsa[centerIndex]=FZERO;
          }

          double tmp = 0.5*(input.h[centerIndex]+temp.hsa[centerIndex]); // combine first and last generated input argument
          if (tmp>=constants.HE_CA){ // blend select with a trick: do the computation anyway and then multiply the result with either 0. or 0.5
            temp.q1[centerIndex] = 0.5*(input.h[centerIndex]*input.u[centerIndex]+temp.hsa[centerIndex]*temp.usa[centerIndex]);
            temp.q2[centerIndex] = 0.5*(input.h[centerIndex]*input.v[centerIndex]+temp.hsa[centerIndex]*temp.vsa[centerIndex]);
          } else{
            temp.q1[centerIndex] = FZERO;
            temp.q2[centerIndex] = FZERO;
          }

          if (tmp>=constants.HE_CA){
            input.h[centerIndex] = tmp;
            input.u[centerIndex] = temp.q1[centerIndex]/tmp; // bad for blend operation? or just perform division and risk NaN?
            input.v[centerIndex] = temp.q2[centerIndex]/tmp; // bad for blend operation? or just perform division and risk NaN?
          } else {
            input.h[centerIndex] = FZERO;
            input.u[centerIndex] = FZERO;
            input.v[centerIndex] = FZERO;
          }
      
          temp.Vin_tot[centerIndex] = (temp.Vin1[centerIndex] + temp.Vin2[centerIndex])*0.5;

          temp.Vin1[centerIndex]=temp.Vin_tot[centerIndex];
          temp.Vin2[centerIndex]=temp.Vin_tot[centerIndex];

          height_Vinf_tot+= temp.Vin_tot[centerIndex];  
          height_of_tot+=input.h[centerIndex];
        } //end for j
      } //end for i

#if 0
      /*Computation of the cumulated infiltrated volume*/
      double Vol_inf_tot_cumul=height_Vinf_tot*DX*DY;

      /*Computation of the overland flow volume*/
      double Vol_of_tot=height_of_tot*DX*DY;

#endif

      tps=tps+dt1;
      n=n+1;

    }//end for else dt2<dt1

    }; // while loop
 
    return dt1; // this is the timestep which was used for this iteration
}


void MekkaFlood_solver::boundary(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants, double time_tmp){

#if 0
    const int NODEX = NXCELL;
    const int NODEY = NYCELL;

    // effectively, if we disable the boundary computation, all result values should become zero, is this good or probably bad?
#if 0
  for (int j=1 ; j<NODEY+1 ; j++){
    Lbound->calcul(h_tmp[1][j],u_tmp[1][j],v_tmp[1][j],L_IMP_H,L_IMP_Q,h_tmp[NODEX][j],u_tmp[NODEX][j],v_tmp[NODEX][j], time_tmp,-1,0);
    h_tmp[0][j] = Lbound->get_hbound();
    u_tmp[0][j] = Lbound->get_unormbound();
    v_tmp[0][j] = Lbound->get_utanbound();
    Rbound->calcul(h_tmp[NODEX][j],u_tmp[NODEX][j],v_tmp[NODEX][j],R_IMP_H,R_IMP_Q,h_tmp[1][j],u_tmp[1][j],v_tmp[1][j], time_tmp,1,0);
    h_tmp[NODEX+1][j] = Rbound->get_hbound();
    u_tmp[NODEX+1][j] = Rbound->get_unormbound();
    v_tmp[NODEX+1][j] = Rbound->get_utanbound();
  } //end for j
  
  for (int i=1 ; i<NODEX+1 ; i++){
    Bbound->calcul(h_tmp[i][1],v_tmp[i][1],u_tmp[i][1],B_IMP_H,B_IMP_Q,h_tmp[i][NODEY],v_tmp[i][NODEY],u_tmp[i][NODEY], time_tmp,0,-1);
    h_tmp[i][0] = Bbound->get_hbound();
    u_tmp[i][0] = Bbound->get_utanbound();
    v_tmp[i][0] = Bbound->get_unormbound();
    
    Tbound->calcul(h_tmp[i][NODEY],v_tmp[i][NODEY],u_tmp[i][NODEY],T_IMP_H,T_IMP_Q,h_tmp[i][1],v_tmp[i][1],u_tmp[i][1], time_tmp,0,1);
    h_tmp[i][NODEY+1] = Tbound->get_hbound();
    u_tmp[i][NODEY+1] = Tbound->get_utanbound();
    v_tmp[i][NODEY+1] = Tbound->get_unormbound();
  } //end for i
#endif
#endif
}


double MekkaFlood_solver::lim_minmod(double a, double b) {
    double rec;
	if (a>=FZERO && b>=FZERO){
		rec=std::min(a,b);
	}else if (a<=FZERO && b<=FZERO){
		rec=std::max(a,b);
	}else{
		rec=FZERO;
	}
    return rec;
}

void MekkaFlood_solver::rec_muscl_init(const int patchid, int dim, unsigned int* strideinfo, InputArrays& input, TempArrays& temp, const Constants& constants) {
    const int NXCELL = constants.NXCELL;
    const int NYCELL = constants.NYCELL;

    unsigned int index[3];
 
    unsigned int bottomIndex;
    unsigned int topIndex;
    unsigned int centerIndex;
    unsigned int leftIndex;
    unsigned int rightIndex;

	for (int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j;
        index[1] = 0;
        index[2] = patchid;
        leftIndex = linearizeIndex(3, index, strideinfo);
 
        index[0] = j;
        index[1] = 1;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);

	    temp.z1r[leftIndex] = input.z[leftIndex];
	    temp.delta_z1[leftIndex] = input.z[centerIndex]-input.z[leftIndex];
 
        index[0] = j;
        index[1] = NXCELL+1;
        index[2] = patchid;
        rightIndex = linearizeIndex(3, index, strideinfo);

	    temp.z1l[rightIndex] = input.z[rightIndex];
	} //end for j


	for (int i=1 ; i<NXCELL+1 ; i++){
        index[0] = 0;
        index[1] = i;
        index[2] = patchid;
        bottomIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = NYCELL+1;
        index[1] = i;
        index[2] = patchid;
        topIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = 1;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
 
	  temp.z2r[bottomIndex] = input.z[bottomIndex];
	  temp.z2l[topIndex] = input.z[topIndex];
	  temp.delta_z2[bottomIndex] = input.z[centerIndex]-input.z[bottomIndex];
	  for (int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
 
        index[0] = j;
        index[1] = i+1;
        index[2] = patchid;
        rightIndex = linearizeIndex(3, index, strideinfo);

        index[0] = j+1;
        index[1] = i;
        index[2] = patchid;
        topIndex = linearizeIndex(3, index, strideinfo);

	    temp.delta_z1[centerIndex] = input.z[rightIndex]-input.z[centerIndex];
	    temp.delta_z2[centerIndex] = input.z[topIndex]-input.z[centerIndex];
	  } //end for j
	} //end for i
}

void MekkaFlood_solver::rec_muscl(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants){
    unsigned int index[3];
    unsigned int topIndex;
    unsigned int centerIndex;
    unsigned int leftIndex;
    unsigned int rightIndex;
    unsigned int bottomIndex;

    const int NXCELL = constants.NXCELL;
    const int NYCELL = constants.NYCELL;

	for(int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j; // >= 1
        index[1] = 0; // i
        index[2] = patchid;
        leftIndex = linearizeIndex(3, index, strideinfo);
	
        index[0] = j;
        index[1] = 1; // i
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
	
		temp.h1r[leftIndex] = input.h[leftIndex];
		temp.u1r[leftIndex] = input.u[leftIndex];
		temp.v1r[leftIndex] = input.v[leftIndex];

		double delta_h1 = input.h[centerIndex] - input.h[leftIndex]; //delta_h1 = h[1][j]-h[0][j];
		double delta_u1 = input.u[centerIndex] - input.u[leftIndex]; //delta_u1 = u[1][j]-u[0][j];
		double delta_v1 = input.v[centerIndex] - input.v[leftIndex]; //delta_v1 = v[1][j]-v[0][j];

		for(int i=1 ; i<NXCELL+1 ; i++){
            index[0] = j;
            index[1] = i+1;
            index[2] = patchid;
            rightIndex = linearizeIndex(3, index, strideinfo);
  
            index[0] = j;
            index[1] = i-1;
            index[2] = patchid;
            leftIndex = linearizeIndex(3, index, strideinfo);
     
            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            centerIndex = linearizeIndex(3, index, strideinfo);

			double delta_h2 = input.h[rightIndex]-input.h[centerIndex]; // delta_h2 = h[i+1][j]-h[i][j];
			double delta_u2 = input.u[rightIndex]-input.u[centerIndex]; // delta_u2 = u[i+1][j]-u[i][j];
            double delta_v2 = input.v[rightIndex]-input.v[centerIndex]; // delta_v2 = v[i+1][j]-v[i][j];

            // TODO: calls to method, let us see how much this actually hurts
            // maybe we have to provide the return value by reference?
			double dh = lim_minmod(delta_h1,delta_h2);
			double dz_h = lim_minmod(delta_h1+temp.delta_z1[leftIndex],delta_h2+temp.delta_z1[centerIndex]);
			double du = lim_minmod(delta_u1,delta_u2);
			double dv = lim_minmod(delta_v1,delta_v2);

			temp.h1r[centerIndex]=input.h[centerIndex]+dh*0.5;
			temp.h1l[centerIndex]=input.h[centerIndex]-dh*0.5;

			temp.z1r[centerIndex]=input.z[centerIndex]+0.5*(dz_h-dh);
			temp.z1l[centerIndex]=input.z[centerIndex]+0.5*(dh-dz_h);
			temp.delzc1[centerIndex] = temp.z1r[centerIndex]-temp.z1l[centerIndex];
			temp.delz1[leftIndex] = temp.z1l[centerIndex]-temp.z1r[leftIndex];

			if (input.h[centerIndex]>0.){
				temp.u1r[centerIndex]=input.u[centerIndex]+temp.h1l[centerIndex]*du*0.5/input.h[centerIndex];
				temp.u1l[centerIndex]=input.u[centerIndex]-temp.h1r[centerIndex]*du*0.5/input.h[centerIndex];
				temp.v1r[centerIndex]=input.v[centerIndex]+temp.h1l[centerIndex]*dv*0.5/input.h[centerIndex];
				temp.v1l[centerIndex]=input.v[centerIndex]-temp.h1r[centerIndex]*dv*0.5/input.h[centerIndex];
			}else{
				temp.u1r[centerIndex]=input.u[centerIndex]+du*0.5;
				temp.u1l[centerIndex]=input.u[centerIndex]-du*0.5;
				temp.v1r[centerIndex]=input.v[centerIndex]+dv*0.5;
				temp.v1l[centerIndex]=input.v[centerIndex]-dv*0.5;
			} //end if

            //rajout de l'iteration des delta
            delta_h1 = delta_h2;
            delta_u1 = delta_u2;
            delta_v1 = delta_v2;
		} //end for i
 
        index[0] = j;
        index[1] = NXCELL+1; // i
        index[2] = patchid;
        rightIndex = linearizeIndex(3, index, strideinfo);
 
        index[0] = j;
        index[1] = NXCELL; // i
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);

		temp.h1l[rightIndex] = input.h[rightIndex];
		temp.u1l[rightIndex] = input.u[rightIndex];
		temp.v1l[rightIndex] = input.v[rightIndex];

		temp.delz1[centerIndex] = temp.z1l[rightIndex]-temp.z1r[centerIndex]; 
	} //end for j
	
    for (int i=1 ; i<NXCELL+1 ; i++){
        index[0] = 1;
        index[1] = i; // >= 1
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = 0;
        index[1] = i;
        index[2] = patchid;
        bottomIndex = linearizeIndex(3, index, strideinfo);
	
        temp.h2r[bottomIndex] = input.h[bottomIndex];
		temp.u2r[bottomIndex] = input.u[bottomIndex];
		temp.v2r[bottomIndex] = input.v[bottomIndex];

		double delta_h1 = input.h[centerIndex]-input.h[bottomIndex];
		double delta_u1 = input.u[centerIndex]-input.u[bottomIndex];
		double delta_v1 = input.v[centerIndex]-input.v[bottomIndex];
		
        for (int j=1 ; j<NYCELL+1 ; j++){

            index[0] = j;
            index[1] = i;
            index[2] = patchid;
            centerIndex = linearizeIndex(3, index, strideinfo);
      
            index[0] = j+1;
            index[1] = i;
            index[2] = patchid;
            topIndex = linearizeIndex(3, index, strideinfo);
  
            index[0] = j-1;
            index[1] = i;
            index[2] = patchid;
            bottomIndex = linearizeIndex(3, index, strideinfo);
      
			double delta_h2 = input.h[topIndex]-input.h[centerIndex];
			double delta_u2 = input.u[topIndex]-input.u[centerIndex];
			double delta_v2 = input.v[topIndex]-input.v[centerIndex];

			double dh = lim_minmod(delta_h1,delta_h2);
			double dz_h = lim_minmod(delta_h1+temp.delta_z2[bottomIndex],delta_h2+temp.delta_z2[centerIndex]);
			double du = lim_minmod(delta_u1,delta_u2);
			double dv = lim_minmod(delta_v1,delta_v2);

			temp.h2r[centerIndex] = input.h[centerIndex]+dh*0.5;
			temp.h2l[centerIndex] = input.h[centerIndex]-dh*0.5;

			temp.z2r[centerIndex] = input.z[centerIndex]+0.5*(dz_h-dh);
			temp.z2l[centerIndex] = input.z[centerIndex]+0.5*(dh-dz_h);
			temp.delzc2[centerIndex] = temp.z2r[centerIndex]-temp.z2l[centerIndex];
			temp.delz2[bottomIndex] = temp.z2l[centerIndex]-temp.z2r[bottomIndex];

            double scalar = 0.0;
			if (input.h[centerIndex]>0.) { // blend / select operation
                scalar = temp.h2l[centerIndex] / input.h[centerIndex];
            } else {
                scalar = 1.0;
            }

            temp.u2r[centerIndex] = input.u[centerIndex]+scalar*du*0.5;
            temp.v2r[centerIndex] = input.v[centerIndex]+scalar*dv*0.5;
  
			if (input.h[centerIndex]>0.) { // blend / select operation
                scalar = temp.h2r[centerIndex] / input.h[centerIndex];
            } else {
                scalar = 1.0;
            }

            temp.u2l[centerIndex] = input.u[centerIndex]-scalar*du*0.5;
            temp.v2l[centerIndex] = input.v[centerIndex]-scalar*dv*0.5;
       
            //rajout de l'iteration des delta
            delta_h1 = delta_h2;
            delta_u1 = delta_u2;
            delta_v1 = delta_v2;

		} //end for j

        index[0] = NYCELL+1;
        index[1] = i; // >= 1
        index[2] = patchid;
        topIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = NYCELL;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
	
		temp.h2l[topIndex] = input.h[topIndex];
		temp.u2l[topIndex] = input.u[topIndex];
		temp.v2l[topIndex] = input.v[topIndex];

		temp.delz2[centerIndex] = temp.z2l[topIndex]-temp.z2r[centerIndex];
	} //end for i
}

void MekkaFlood_solver::rec_hydro(double hg,double hd,double dz, double& hg_rec, double& hd_rec){
	
	/**
	 * @details
	 * See \cite Audusse04c for more details.
	 * @param[in] hg water height on the cell located at the left of the boundary.
	 * @param[in] hd water height on the cell located at the right of the boundary.
	 * @param[in] dz Difference between the values of the topography of the two adjacent cells.
	 * @par Modifies
	 * Hydrostatic#hl_rec, set to \f$ \left(hl\_0 -\max (0, dz) \right)_+\f$.\n
	 * Hydrostatic#hr_rec, set to \f$ \left(hr\_0 -\max (0, -dz) \right)_+\f$.
	 */
	
  hg_rec = std::max(FZERO,hg-std::max(FZERO,dz));
  hd_rec = std::max(FZERO,hd-std::max(FZERO,-dz));
}

void MekkaFlood_solver::flux_hll(const Constants& constants, double h_L,double u_L,double v_L,double h_R,double u_R,double v_R, double& f1, double& f2, double& f3, double& cfl){
	if (h_L<=0. && h_R<=0.){
		f1 = 0.;
		f2 = 0.;
		f3 = 0.;
		cfl = 0.;
	}else{
		double grav_h_L = constants.GRAVITATION*h_L;
		double grav_h_R = constants.GRAVITATION*h_R;
		double q_R = u_R*h_R;
		double q_L = u_L*h_L;
		double c1 = std::min(u_L-sqrt(grav_h_L),u_R-sqrt(grav_h_R));
		double c2 = std::max(u_L+sqrt(grav_h_L),u_R+sqrt(grav_h_R));

		//cfl is the velocity to calculate the real cfl=max(fabs(c1),fabs(c2))*tx with tx=dt/dx
        if (fabs(c1)<constants.EPSILON && fabs(c2)<constants.EPSILON){              //dry state
			f1=0.;
			f2=0.;
			f3=0.;
			cfl=0.; //max(fabs(c1),fabs(c2))=0
		}else if (c1>=constants.EPSILON){ //supercritical flow, from left to right : we have max(abs(c1),abs(c2))=c2>0 
			f1=q_L;
			f2=q_L*u_L+constants.GRAVITATION*h_L*h_L*0.5;
			f3=q_L*v_L;
			cfl=c2; //max(fabs(c1),fabs(c2))=c2>0
		}else if (c2<=-constants.EPSILON){ //supercritical flow, from right to left : we have max(abs(c1),abs(c2))=-c1>0
			f1=q_R;
			f2=q_R*u_R+constants.GRAVITATION*h_R*h_R*0.5;
			f3=q_R*v_R;
			cfl=fabs(c1); //max(fabs(c1),fabs(c2))=fabs(c1)
		}else{ //subcritical flow
		        double tmp = 1./(c2-c1);
			f1=(c2*q_L-c1*q_R)*tmp+c1*c2*(h_R-h_L)*tmp;
			f2=(c2*(q_L*u_L+constants.GRAVITATION*h_L*h_L*0.5)-c1*(q_R*u_R+constants.GRAVITATION*h_R*h_R*0.5))*tmp+c1*c2*(q_R-q_L)*tmp;
			f3=(c2*(q_L*v_L)-c1*(q_R*v_R))*tmp+c1*c2*(h_R*v_R-h_L*v_L)*tmp;
			cfl=std::max(fabs(c1),fabs(c2));
		}
	}
}

// this method originally used an variable "T" which specified the maximum simulation time.
// However, this variable is not used anymore. Similar holds for "tps"
void MekkaFlood_solver::maincalcflux(const int patchid, int dim, unsigned int* strideinfo, TempArrays& temp, const Constants& constants,
                                     double cflfix, double dt_max, double& dt) { // dt is both input and output
  double dt_tmp,dtx,dty;
  double velocity_max_x,velocity_max_y;              //tempory velocity to verify if clf > cflfix
  dtx=dty=dt_max;
  velocity_max_x=velocity_max_y=-constants.VE_CA;
 
  const int NXCELL = constants.NXCELL;
  const int NYCELL = constants.NYCELL;

  unsigned int index[3];
  unsigned int leftIndex;
  unsigned int centerIndex;
  unsigned int bottomIndex;
 
  double cfl = 0.0;

  for (int i=1 ; i<=NXCELL+1 ; i++){
    for (int j=1 ; j<NYCELL+1 ; j++){
 
        index[0] = j;
        index[1] = i-1;
        index[2] = patchid;
        leftIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
  
      rec_hydro(temp.h1r[leftIndex],temp.h1l[centerIndex],temp.delz1[leftIndex], temp.h1right[leftIndex], temp.h1left[centerIndex]);
      flux_hll(constants, temp.h1right[leftIndex],temp.u1r[leftIndex],temp.v1r[leftIndex],temp.h1left[centerIndex],temp.u1l[centerIndex],temp.v1l[centerIndex], temp.f1[centerIndex], temp.f2[centerIndex], temp.f3[centerIndex], cfl);
      //f1[i][j] = flux_num->get_f1();
      //f2[i][j] = flux_num->get_f2();
      //f3[i][j] = flux_num->get_f3();

      if (fabs(cfl*dt/constants.DX) < constants.EPSILON) { // good candidate for blend / select operation
	    dt_tmp=dt_max;
      }else{
	    //	dt_tmp=min(T-tps,cflfix*DX/flux_num->get_cfl());
	    dt_tmp=cflfix*constants.DX/cfl;
      }
      dtx = std::min(std::min(dt,dt_tmp),dtx);
      velocity_max_x = std::max(velocity_max_x,cfl);

    } //end for j
  } //end for i
 
  std::cout << "- dtx " << dtx << std::endl;


  for (int i=1 ; i<NXCELL+1 ; i++){
    for (int j=1 ; j<=NYCELL+1 ; j++){

        index[0] = j-1;
        index[1] = i;
        index[2] = patchid;
        bottomIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
 
      rec_hydro(temp.h2r[bottomIndex],temp.h2l[centerIndex],temp.delz2[bottomIndex], temp.h2right[bottomIndex], temp.h2left[centerIndex]);

      flux_hll(constants, temp.h2right[bottomIndex],temp.v2r[bottomIndex],temp.u2r[bottomIndex],temp.h2left[centerIndex],temp.v2l[centerIndex],temp.u2l[centerIndex],
               temp.g1[centerIndex], temp.g3[centerIndex], temp.g2[centerIndex], cfl );
      //g1[i][j] = flux_num->get_f1();
      //g2[i][j] = flux_num->get_f3(); // this looks wrong but the arguments to the flux are also twisted
      //g3[i][j] = flux_num->get_f2();


      if (fabs(cfl*dt/constants.DY) < constants.EPSILON) { // good candidate for blend / select operation
        dt_tmp=dt_max;
      }else{
        dt_tmp=cflfix*constants.DY/cfl;
      }
      dty = std::min(std::min(dt,dt_tmp),dty);
      velocity_max_y = std::max(velocity_max_y,cfl);
    
    } //end for j
  } //end for i
 
    std::cout << "- dty " << dty << std::endl;

#if 0 // TODO: this might be changeable in the future, but for this DEMO case the SCHEME_TYPE is always 1

  if (1 == SCHEME_TYPE){
    dt_cal=std::min(dtx,dty);
    //cout << " computed new timestep: " << dt_cal << "and got cfl " << flux_num->get_cfl() << endl;
  }else{
      double x_lhs = velocity_max_x*DT_FIX/constants.DX;
      double y_lhs = velocity_max_y*DT_FIX/constants.DY>cflfix;
    if ((x_lhs/DX>cflfix)||(y_lhs>cflfix)){
      cout << " the CFL condition is not satisfied: CFL >"<<cflfix << " " << x_lhs << " " << y_lhs << " " << DX << " " << DY << endl;
      exit(1);
    } //end if
    dt_cal=DT_FIX;    
  }

#else
    // dt_cal is now dt
    dt= std::min(dtx,dty);

#endif 
}

void MekkaFlood_solver::maincalcscheme(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, SchemeArrays& output, TempArrays& temp, const Constants& constants,
                                       double tps, double dt, int verif)
{

    const int NXCELL = constants.NXCELL;
    const int NYCELL = constants.NYCELL;
    
    unsigned int index[3];
    unsigned int centerIndex;
    unsigned int rightIndex;
    unsigned int topIndex;

/*------------------------Periodic boundary condition----------------------------------------------------------*/
#if 0
  //In case of periodic boundary condition  it's necessary that the flux at the boundary (Left and Right, Bottom and Top) are the same.
  // are the same.
  //moreover we need to consider the direction of the discharge in order to exchange the flux.
    if ((4==nchoice_Tbound) && (4==nchoice_Bbound)) {
    for (int i=1 ; i<NXCELL+1 ; i++){
      if ((ve2[i][1] > 0.) && (ve2[i][NYCELL]> 0.)){ //the direction of flow is Bottom to the Top
	g1[i][1]= g1[i][NYCELL+1];
      }
      if ((ve2[i][1] < 0.) && (ve2[i][NYCELL]< 0.)){ //the direction of flow is Top to the Bottom
	g1[i][NYCELL+1]	= g1[i][1];
      }
    }
  } 
    
  if ((4==nchoice_Rbound) && (4==nchoice_Lbound)) {
    for (int j=1 ; j<NYCELL+1 ; j++){
      if ((ve1[1][j] > 0.) && (ve1[NXCELL][j]> 0.)){ //the direction of flow is Left to the Right
	f1[1][j]= f1[NXCELL+1][j];
      }
    }
    for (int j=1 ; j<NYCELL+1 ; j++){
      if ((ve1[1][j] < 0.) && (ve1[NXCELL][j]< 0.)){ //the direction of flow is Right to the Left
	f1[NXCELL+1][j] = f1[1][j];
      }
    }
  }
#endif

  /*---------------------------------------------------------------------------------------------------------*/    
 //Rainfall and infiltration calculated in Saint-Venant system
  //  rain(tps,P);  
  rain(patchid, dim, strideinfo, input, temp, constants, tps);
                        
  /*---------------------------------------------------------------------------------------------------------*/    

  double tx=dt/constants.DX;
  double ty=dt/constants.DY;
  
  double dt_first = FZERO;

  // flux' s computation  at each boundaries-----------------------------------------------
    if (1 == verif){

        dt_first=dt;

     }
     

  // --------------------------------------------------------------------------------------
  
 for (int i=1 ; i<NXCELL+1 ; i++){
   for (int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j;
        index[1] = i+1;
        index[2] = patchid;
        rightIndex = linearizeIndex(3, index, strideinfo);
 
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = j+1;
        index[1] = i;
        index[2] = patchid;
        topIndex = linearizeIndex(3, index, strideinfo);

     // Solution of the equation of mass conservation (First equation of Saint venant)
     output.h[centerIndex] = input.h[centerIndex]-tx*(temp.f1[rightIndex]-temp.f1[centerIndex])
                                                 -ty*(temp.g1[topIndex]-temp.g1[centerIndex])+temp.Tab_rain[centerIndex]*dt;
       
   } //end for j
 } //end for i	
 
 //------------------------------------------

 
 //Infiltration case  ------------------------  
 // though it currently does nothing (TODO): this has a temp array which gets copied back later
 infiltration(patchid, dim, strideinfo, input, temp, constants, dt);

 //------------------------------------------
 
 for (int i=1 ; i<NXCELL+1 ; i++){
   for (int j=1 ; j<NYCELL+1 ; j++){
     //------------------------------------------
      
        index[0] = j;
        index[1] = i+1;
        index[2] = patchid;
        rightIndex = linearizeIndex(3, index, strideinfo);
 
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
  
        index[0] = j+1;
        index[1] = i;
        index[2] = patchid;
        topIndex = linearizeIndex(3, index, strideinfo);

     if (output.h[centerIndex] > constants.HE_CA){ // TODO; blend /select is a little bit difficult here
                                  // maybe: just do the computation and the keep old q1 / q2 or update it with new (valid) values
       //Solution of the equation of momentum (Second and third equation of Saint-venant)
       
       output.q1[centerIndex] = input.h[centerIndex]*input.u[centerIndex]-tx*(temp.f2[rightIndex]-temp.f2[centerIndex]
                                                        +constants.GRAVITATION_DEM*((temp.h1left[centerIndex]-temp.h1l[centerIndex])*(temp.h1left[centerIndex]+temp.h1l[centerIndex])
                                                                  +(temp.h1r[centerIndex]-temp.h1right[centerIndex])*(temp.h1r[centerIndex]+temp.h1right[centerIndex])
                                                                  +(temp.h1l[centerIndex]+temp.h1r[centerIndex])*temp.delzc1[centerIndex]))
                                                    -ty*(temp.g2[topIndex]-temp.g2[centerIndex]);
       output.q2[centerIndex] = input.h[centerIndex]*input.v[centerIndex]-tx*(temp.f3[rightIndex]-temp.f3[centerIndex])
                                                    -ty*(temp.g3[topIndex]-temp.g3[centerIndex]
                                                        +constants.GRAVITATION_DEM*((temp.h2left[centerIndex]-temp.h2l[centerIndex])*(temp.h2left[centerIndex]+temp.h2l[centerIndex])
                                                                   +(temp.h2r[centerIndex]-temp.h2right[centerIndex])*(temp.h2r[centerIndex]+temp.h2right[centerIndex])
                                                                   +(temp.h2l[centerIndex]+temp.h2r[centerIndex])*temp.delzc2[centerIndex]));
       
       //Calcul friction in semi-implicit.
       //TODO: q1 and q2 is source and destination as well? howto solve? e.g. q1 - > temp1 -> q1
       friction(input.u[centerIndex],input.v[centerIndex],output.h[centerIndex],output.q1[centerIndex],output.q2[centerIndex],dt,
                temp.Fric_tab[centerIndex], output.q1[centerIndex], output.q2[centerIndex]);
       
       output.u[centerIndex] = output.q1[centerIndex]/output.h[centerIndex];
       output.v[centerIndex] = output.q2[centerIndex]/output.h[centerIndex];
     }else{ // Case of height of water is zero.
       output.u[centerIndex] = 0.;
       output.v[centerIndex] = 0.;
       
     }				
   } //end for j
 } //end for i	
 
#if 0 // TODO: we might not need this information for our demo case
 /*--------------------*/
 
 // The total cumulated rain's computed
 for (int i=1 ; i<NXCELL+1 ; i++){
   for (int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
 
#if 0  // TODO: we have a special demo case: ORDER = 2
     Volrain_Tot += temp.Tab_rain[centerIndex]*(dt-dt_first*(1-verif))*(1./ORDER)*constants.DX*constans.DY;
#else
     Volrain_Tot += temp.Tab_rain[centerIndex]*(dt-dt_first*(1-verif))*(0.5)*constants.DX*constants.DY;
#endif
   }
 }
#endif

 /*--------------------*/
}

void MekkaFlood_solver::rain(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants,
                             double time)
{
  const int NXCELL = constants.NXCELL;
  const int NYCELL = constants.NYCELL;

  unsigned int index[3];
  unsigned int centerIndex;

  for (int i=1 ; i<NXCELL+1 ; i++){
    for (int j=1 ; j<NYCELL+1 ; j++){
        index[0] = j;
        index[1] = i;
        index[2] = patchid;
        centerIndex = linearizeIndex(3, index, strideinfo);
 

      temp.Tab_rain[centerIndex]=constants.RainIntensity;
    }
  }
}

void MekkaFlood_solver::friction(double uold, double vold, double hnew, double q1new, double q2new, double dt, double cf, double& q1mod, double& q2mod)
{
#if 0 // really do nothing
	q1mod = q1new;
	q2mod = q2new;
#endif
}

void MekkaFlood_solver::infiltration(const int patchid, int dim, unsigned int* strideinfo, SchemeArrays& input, TempArrays& temp, const Constants& constants,
                                     double dt)
{
#if 0 // really do nothing
  for (int i=1 ; i<NXCELL+1 ; i++){
    for (int j=1 ; j<NYCELL+1 ; j++){
      hmod_temp[i][j] = h[i][j]; // water height is no modified.
      Vin_temp[i][j] = Vin_tot[i][j]; // total infiltrated volum is no modified.     
    } //end for j
  } //end for i

  // copy temp result to destination
  h = hmod_temp;
  Vin_tot  = Vin_temp;
#endif
}  
