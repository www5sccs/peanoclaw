/*
 * NumberOfUnknownsDispatcher.h
 *
 *  Created on: May 7, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_NUMBEROFUNKNOWNSDISPATCHER_H_
#define PEANOCLAW_GRID_NUMBEROFUNKNOWNSDISPATCHER_H_

#define dispatchNumberOfUnknowns(numberOfUnknowns, type, methodCall) \
{ \
  switch(numberOfUnknowns) { \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    case 1: \
      { \
        type<1> type1; \
        type1.methodCall; \
      } \
    default: \
      assertionFail("Number of Unknowns " << numberOfUnknowns << " not supported!"); \
    } \


#endif /* PEANOCLAW_GRID_NUMBEROFUNKNOWNSDISPATCHER_H_ */
