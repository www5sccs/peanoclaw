/*
 * Heap.h
 *
 *  Created on: Sep 24, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_HEAP_H_
#define PEANOCLAW_HEAP_H_

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"
#include "peanoclaw/statistics/LevelStatistics.h"
#include "peanoclaw/statistics/ProcessStatisticsEntry.h"
#include "peanoclaw/statistics/TimeIntervalStatistics.h"
#include "peanoclaw/records/PatchDescription.h"
#include "peanoclaw/records/VertexDescription.h"

#include "peano/heap/Heap.h"

//#define noPackedEmptyHeapMessages

#ifdef noPackedEmptyHeapMessages
typedef peano::heap::PlainHeap<peanoclaw::records::CellDescription> CellDescriptionHeap;
typedef peano::heap::PlainHeap<peanoclaw::records::Data> DataHeap;
typedef peano::heap::PlainHeap<peanoclaw::statistics::LevelStatistics> LevelStatisticsHeap;
typedef peano::heap::PlainHeap<peanoclaw::records::PatchDescription> PatchDescriptionHeap;
typedef peano::heap::PlainHeap<peanoclaw::statistics::TimeIntervalStatistics> TimeIntervalStatisticsHeap;
typedef peano::heap::PlainHeap<peanoclaw::statistics::ProcessStatisticsEntry> ProcessStatisticsHeap;
#else
typedef peano::heap::RLEHeap<peanoclaw::records::CellDescription> CellDescriptionHeap;
typedef peano::heap::RLEHeap<peanoclaw::records::Data> DataHeap;
typedef peano::heap::RLEHeap<peanoclaw::statistics::LevelStatistics> LevelStatisticsHeap;
typedef peano::heap::RLEHeap<peanoclaw::records::PatchDescription> PatchDescriptionHeap;
typedef peano::heap::RLEHeap<peanoclaw::statistics::TimeIntervalStatistics> TimeIntervalStatisticsHeap;
typedef peano::heap::RLEHeap<peanoclaw::statistics::ProcessStatisticsEntry> ProcessStatisticsHeap;
#endif

#endif /* PEANOCLAW_HEAP_H_ */
