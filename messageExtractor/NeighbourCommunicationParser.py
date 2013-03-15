import re
from Message import Message
from Parser import Parser

class NeighbourCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareSendToNeighbourPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToNeighbour\(...\)\s*in:.*toRank:(\d+)")
    self.mergeWithNeighbourPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithNeighbour\(...\)\s*in:.*fromRank:(\d)")
    
  def parseLine(self, line):
    m = self.prepareSendToNeighbourPattern.search(line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(2)
      message = Message("PrepareSendToNeighbour")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      return message
    else:
      m = self.mergeWithNeighbourPattern.search(line)
      if m:
        toRank = m.group(1)
        fromRank = m.group(2)
        message = Message("MergeWithNeighbour")
        message.addAttribute("From", fromRank)
        message.addAttribute("To", toRank)
        return message        
      else:
        return None
      
      