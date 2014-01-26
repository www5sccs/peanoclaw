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
    
  times[rank].time += time
  times[rank].samples += samples
  
  print "samples:" + str(times[rank].samples)
  
def processLine(line, times):
  rank = int(re.search("rank:(\d*)", line).group(1))
  time = float(re.search("(" + FLOAT_PATTERN + ") \(total\)", line).group(1))
  samples = int(re.search("(\d*) samples", line).group(1))
  
  addTime(times, rank, time, samples)

def main():
  masterWorkerSpacetreeTimes = dict()
  masterWorkerSubgridTimes = dict()
  neighborSubgridTimes = dict()

  for line in sys.stdin:
    if re.search("Waiting time for master-worker spacetree", line):
      processLine(line, masterWorkerSpacetreeTimes)
      
    if re.search("Waiting time for master-worker subgrid", line):
      processLine(line, masterWorkerSubgridTimes)
      
    if re.search("Waiting time for neighbor subgrid", line):
      processLine(line, neighborSubgridTimes)
      
  ranks = masterWorkerSubgridTimes.keys()
  ranks.sort()
  for rank in ranks:
    print str(rank) + ",\t" + str(masterWorkerSpacetreeTimes[rank].time) + ",\t" + str(masterWorkerSubgridTimes[rank].time) + ",\t" + str(neighborSubgridTimes[rank].time) + ",\t" + str(masterWorkerSpacetreeTimes[rank].samples)
 
if __name__ == "__main__":
    main()
