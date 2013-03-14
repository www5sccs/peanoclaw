import re
from Parser import Parser
from Message import Message

class LoadBalancingMessageParser(Parser):

  def __init__(self):  
    self.pattern = "rank:(\d+) peano::grid::nodes::Node::updateCellsParallelStateAfterLoadForRootOfDeployedSubtree\(\)         send balancing message \(loadBalancingFlag:(-?\d+)\) to rank (\d+)"
    
  def parseLine(self, line):
    m = re.search(self.pattern, line)
    if m:
      fromRank = m.group(1)
      loadbalancingFlag = m.group(2) 
      toRank = m.group(3)
      
      message = Message("LoadBalancingMessage")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("flag", loadbalancingFlag)

      return message
    else:
      return None
    
    