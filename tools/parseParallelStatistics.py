import sys
import re

FLOAT_PATTERN = "[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"

class Entry:
  def __init__(self, time, samples):
    self.time = time
    self.samples = samples

def addTime(times, rank, time, samples):
  if not times.has_key(rank):
    times[rank] = Entry(0.0, 0)
    
  #times[rank].time += time
  #times[rank].samples += samples
  times[rank].time = time
  times[rank].samples = samples

  
def processLine(line, times):
  rank = int(re.search("rank:(\d*)", line).group(1))
  time = float(re.search("(" + FLOAT_PATTERN + ") \(total\)", line).group(1))
  samples = int(re.search("(\d*) samples", line).group(1))
  
  addTime(times, rank, time, samples)

def main():
  masterWorkerSpacetreeTimes = dict()
  masterWorkerSubgridTimes = dict()
  neighborSubgridTimes = dict()
  neighborSpacetreeTimes = dict()
  verticalSpacetreeTimes = dict()
  sendCellAndVerticesToMasterTimes = dict()
  sendStateToMasterTimes = dict()
  startupTimes = dict()
  beginIterationTimes = dict()
  iterationTimes = dict()

  for line in sys.stdin:
    if re.search("Waiting time for master-worker spacetree", line):
      processLine(line, masterWorkerSpacetreeTimes)
      
    if re.search("Waiting time for master-worker subgrid", line):
      processLine(line, masterWorkerSubgridTimes)
      
    if re.search("Waiting time for neighbor subgrid", line):
      processLine(line, neighborSubgridTimes)
      
    if re.search("Waiting time for neighbor spacetree", line):
      processLine(line, neighborSpacetreeTimes)
      
    if re.search("Waiting time for vertical spacetree", line):
      processLine(line, verticalSpacetreeTimes)
      
    if re.search("Waiting time for sending cell and vertices to master", line):
      processLine(line, sendCellAndVerticesToMasterTimes)
      
    if re.search("Waiting time for sending state to master", line):
      processLine(line, sendStateToMasterTimes)
      
    if re.search("Waiting time for startup", line):
      processLine(line, startupTimes)
      
    if re.search("Waiting time for begin iteration", line):
      processLine(line, beginIterationTimes)
      
    if re.search("Waiting time for iteration", line):
      processLine(line, iterationTimes)
      
  #Header
  print "#Rank\tm-w s/t\t\tm-w s/g\t\tn s/g\t\tn s/t\t\tv s/t\t\tit"
      
  ranks = masterWorkerSubgridTimes.keys()
  ranks.sort()
  for rank in ranks:
    neighborSpacetreeTime = "0\t"
    if neighborSpacetreeTimes.has_key(rank):
      neighborSpacetreeTime = "%.2e" % (neighborSpacetreeTimes[rank].time)
      
    verticalSpacetreeTime = "0\t"
    if verticalSpacetreeTimes.has_key(rank):
      verticalSpacetreeTime = "%.2e" % (verticalSpacetreeTimes[rank].time)
      
    sendCellAndVerticesToMasterTime = "\t"
    if sendCellAndVerticesToMasterTimes.has_key(rank):
      sendCellAndVerticesToMasterTime = "%.2e" % (sendCellAndVerticesToMasterTimes[rank].time)
      
    sendStateToMasterTime = "0\t"
    if sendStateToMasterTimes.has_key(rank):
      sendStateToMasterTime = "%.2e" % (sendStateToMasterTimes[rank].time)
      
    print str(rank) + ",\t" \
      + "%.2e" % (masterWorkerSpacetreeTimes[rank].time) + ",\t" \
      + "%.2e" % (masterWorkerSubgridTimes[rank].time) + ",\t" \
      + "%.2e" % (neighborSubgridTimes[rank].time) + ",\t" \
      + neighborSpacetreeTime + ",\t" \
      + verticalSpacetreeTime + ",\t" \
      + sendCellAndVerticesToMasterTime + ",\t" \
      + sendStateToMasterTime + ",\t" \
      + "%.2e" % (startupTimes[rank].time) + ",\t" \
      + "%.2e" % (beginIterationTimes[rank].time) + ",\t" \
      + "%.2e" % (iterationTimes[rank].time) + ",\t" \
      + str(masterWorkerSpacetreeTimes[rank].samples)
 
if __name__ == "__main__":
    main()
