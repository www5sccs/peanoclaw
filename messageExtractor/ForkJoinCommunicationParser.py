import re
from Message import Message
from Parser import Parser

class ForkJoinCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareCopyToRemoteNodePattern = "rank:(\d+) peanoclaw::mappings::Remesh::prepareCopyToRemoteNode\(...\)                 in:local(Vertex|Cell).*toRank:(\d+)"
    self.mergeWithRemoteDataDueToForkOrJoinPattern = "rank:(\d+) peanoclaw::mappings::SolveTimestep::mergeWithRemoteDataDueToForkOrJoin\(...\)             in:local(Vertex|Cell).*fromRank:(\d+)"

  def parseLine(self, line):
    m = re.search(self.prepareCopyToRemoteNodePattern, line)
    if m:
      fromRank = m.group(1)
      type = m.group(2)
      toRank = m.group(3)
      message = Message("PrepareCopyToRemoteNode")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("type", type)
      return message
    else:
      m = re.search(self.mergeWithRemoteDataDueToForkOrJoinPattern, line)
      if m:
        toRank = m.group(1)
        type = m.group(2)
        fromRank = m.group(3)
        message = Message("MergeWithRemoteDataDueToForkOrJoin")
        message.addAttribute("From", fromRank)
        message.addAttribute("To", toRank)
        message.addAttribute("type", type)
        return message        
      else:
        return None