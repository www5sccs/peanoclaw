import re
from Parser import Parser
from Message import Message

class StateMessageParser(Parser):
  
  def __init__(self):
    self.sendPattern =  re.compile("rank:(\d+) peano::grid::nodes::Node::updateCellsParallelStateAfterLoadForRootOfDeployedSubtree\(\)\s*send state.*loadRebalancingState:(.*),reduceStateAndCell:\d+\) to rank (\d+)")
    self.receivePattern = re.compile("rank:(\d+) peano::grid::Grid::receiveStartupDataFromMaster\(\)\s*received state")
    
  def parseLine(self, line):
    m = self.sendPattern.search(line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(3)
      
      message = Message("SendState")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      
      return message
    else:
      m = self.receivePattern.search(line)
      if m:
        toRank = m.group(1)
        
        message = Message("ReceivedState")
        message.addAttribute("To", toRank)
        
        return message        
      else:
        return None
      
      