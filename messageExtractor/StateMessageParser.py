import re
from Parser import Parser
from Message import Message

class StateMessageParser(Parser):
  
  def __init__(self):
    self.pattern =  "rank:(\d+) peano::grid::nodes::Node::updateCellsParallelStateAfterLoadForRootOfDeployedSubtree\(\)\s*"
    self.pattern += "send state.*loadRebalancingState:(.*),reduceStateAndCell:\d+\) to rank (\d+)"
    
  def parseLine(self, line):
    m = re.search(self.pattern, line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(3)
      
      message = Message("State")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      
      return message
    else:
      return None
      
      