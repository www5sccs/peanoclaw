/*
 * VectorOperations.h
 *
 *  Created on: Dec 4, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_TARCH_LA_ADDITIONALVECTOROPERATIONS_H_
#define PEANOCLAW_TARCH_LA_ADDITIONALVECTOROPERATIONS_H_

namespace tarch {
  namespace la {
    /**
     * Calculates the area of the hyperplane that is the projection along the projection
     * dimension of the hyper-hexahedron spanned by the given vector.
     */
    template<int Size, typename Scalar>
    Scalar projectedArea(const tarch::la::Vector<Size,Scalar>& vector, int projectionDimension);
  }
}

#include "tarch/la/AdditionalVectorOperations.cpph"

#endif /* PEANOCLAW_TARCH_LA_ADDITIONALVECTOROPERATIONS_H_ */
