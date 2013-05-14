import re
from Message import Message
from Parser import Parser

class ForkJoinCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareCopyToRemoteNodePattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareCopyToRemoteNode\(...\)\s*in:local(Vertex|Cell).*toRank:(\d+),cellCentre:(.*),cellSize:.*,level:(\d+)")
    self.mergeWithRemoteDataDueToForkOrJoinPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithRemoteDataDueToForkOrJoin\(...\)\s*in:local(Vertex|Cell).*fromRank:(\d+),cellCentre:(.*),cellSize:.*,level:(\d+)")

  def parseLine(self, line):
    m = self.prepareCopyToRemoteNodePattern.search(line)
    if m:
      fromRank = m.group(1)
      type = m.group(2)
      toRank = m.group(3)
      position = m.group(4)
      level = m.group(5)
      message = Message("PrepareCopyToRemoteNode")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("Type", type)
      message.addAttribute("Level", level)
      message.addAttribute("Position", position)
      return message
    else:
      m = self.mergeWithRemoteDataDueToForkOrJoinPattern.search(line)
      if m:
        toRank = m.group(1)
        type = m.group(2)
        fromRank = m.group(3)
        position = m.group(4)
        level = m.group(5)
        message = Message("MergeWithRemoteDataDueToForkOrJoin")
        message.addAttribute("From", fromRank)
        message.addAttribute("To", toRank)
        message.addAttribute("Type", type)
        message.addAttribute("Level", level)
        message.addAttribute("Position", position)
        return message        
      else:
        return None
      
      