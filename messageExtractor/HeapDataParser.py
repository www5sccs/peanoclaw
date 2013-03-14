import re
from Parser import Parser
from Message import Message

class HeapDataParser(Parser):
  
  def __init__(self):
    self.heapDataPattern = "rank:(\d+) peano::heap::Heap::sendData                             Sending data at \[.*,.*\] to Rank (\d+) with mpiTag: (\d+)"
    self.neighbourMPITagPattern = "assigned message heap\[neighbour\] the free tag (\d+)"
    self.masterWorkerMPITagPattern = "assigned message heap\[master-worker\] the free tag (\d+)"
    self.forkJoinMPITagPattern = "assigned message heap\[join/fork\] the free tag (\d+)"
    self.neighbourMPITags = []
    self.masterWorkerMPITags = []
    self.forkJoinMPITags = []

  def parseLine(self, line):
    m = re.search(self.heapDataPattern, line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(2)
      mpiTag = m.group(3)
      
      if mpiTag in self.neighbourMPITags:
        mpiTag = "neighbourCommunication"
      elif mpiTag in self.masterWorkerMPITags:
        mpiTag = "MasterWorkerCommunication"
      elif mpiTag in self.forkJoinMPITags:
        mpiTag = "ForkJoinCommunication"
      
      message = Message("HeapData")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("Tag", mpiTag)
      
      #print "Found heap data message [" + fromRank + "," + toRank + "," + mpiTag + "]"
      
      return message
    else:
      m = re.search(self.neighbourMPITagPattern, line)
      if m:
        #print "Found neighbour tag: " + m.group(1)
        self.neighbourMPITags.append(m.group(1))
      m = re.search(self.masterWorkerMPITagPattern, line)
      if m:
        #print "Found master-worker tag: " + m.group(1)
        self.masterWorkerMPITags.append(m.group(1))
      m = re.search(self.forkJoinMPITagPattern, line)
      if m:
        #print "Found fork/join tag: " + m.group(1)
        self.forkJoinMPITags.append(m.group(1))
      
      return None
    