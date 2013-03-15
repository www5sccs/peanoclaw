import re
from Parser import Parser
from Message import Message

class HeapDataParser(Parser):
  
  def __init__(self):
    self.heapDataPattern = re.compile("rank:(\d+) peano::heap::Heap::sendData\s*Sending data at (.*) on level (\d+) to Rank (\d+) with mpiTag: (\d+)")
    self.neighbourMPITagPattern = re.compile("assigned message heap\[neighbour\] the free tag (\d+)")
    self.masterWorkerMPITagPattern = re.compile("assigned message heap\[master-worker\] the free tag (\d+)")
    self.forkJoinMPITagPattern = re.compile("assigned message heap\[join/fork\] the free tag (\d+)")
    self.neighbourMPITags = []
    self.masterWorkerMPITags = []
    self.forkJoinMPITags = []

  def parseLine(self, line):
    m = self.heapDataPattern.search(line)
    if m:
      fromRank = m.group(1)
      position = m.group(2)
      level = m.group(3)
      toRank = m.group(4)
      mpiTag = m.group(5)
      
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
      message.addAttribute("Level", level)
      message.addAttribute("Position", position)
      
      #print "Found heap data message [" + fromRank + "," + toRank + "," + mpiTag + "]"
      
      return message
    else:
      m = self.neighbourMPITagPattern.search(line)
      if m:
        #print "Found neighbour tag: " + m.group(1)
        self.neighbourMPITags.append(m.group(1))
      m = self.masterWorkerMPITagPattern.search(line)
      if m:
        #print "Found master-worker tag: " + m.group(1)
        self.masterWorkerMPITags.append(m.group(1))
      m = self.forkJoinMPITagPattern.search(line)
      if m:
        #print "Found fork/join tag: " + m.group(1)
        self.forkJoinMPITags.append(m.group(1))
      
      return None
    