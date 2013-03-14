import re
from Message import Message
from Parser import Parser

class NeighbourCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareSendToNeighbourPattern = "rank:(\d+) peanoclaw::mappings::InitialiseGrid::prepareSendToNeighbour\(...\).*toRank:(\d+)"
    self.mergeWithNeighbourPattern = "rank:(\d+) peanoclaw::mappings::SolveTimestep::mergeWithNeighbour\(...\).*fromRank:(\d)"
    
  def parseLine(self, line):
    m = re.search(self.prepareSendToNeighbourPattern, line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(2)
      message = Message("PrepareSendToNeighbour")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      return message
    else:
      m = re.search(self.mergeWithNeighbourPattern, line)
      if m:
        toRank = m.group(1)
        fromRank = m.group(2)
        message = Message("MergeWithNeighbour")
        message.addAttribute("From", fromRank)
        message.addAttribute("To", toRank)
        return message        
      else:
        return None
      
      