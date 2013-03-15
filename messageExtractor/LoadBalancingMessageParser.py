import re
from Parser import Parser
from Message import Message

class LoadBalancingMessageParser(Parser):

  def __init__(self):  
    self.sendPattern = re.compile("rank:(\d+) peano::grid::nodes::Node::updateCellsParallelStateAfterLoadForRootOfDeployedSubtree\(\)\s*send balancing message \(loadBalancingFlag:(-?\d+)\) to rank (\d+)")
    self.receivePattern = re.compile("rank:(\d+) peano::parallel::messages::LoadBalancingMessage::receive\(int,int\)\s*received \(loadBalancingFlag:(\d+)\)")
    
  def parseLine(self, line):
    m = self.sendPattern.search(line)
    if m:
      fromRank = m.group(1)
      loadbalancingFlag = m.group(2) 
      toRank = m.group(3)
      
      if loadbalancingFlag == "-2":
        loadbalancingFlag = "Continue"
      elif loadbalancingFlag == "-1":
        loadbalancingFlag = "Join"
      elif loadbalancingFlag == "0":
        loadbalancingFlag = "Undefined"
      elif loadbalancingFlag == "1":
        loadbalancingFlag = "ForkOnce"
      elif int(loadbalancingFlag) > 1:
        loadbalancingFlag = "ForkGreedy"
      
      message = Message("LoadBalancingMessage")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("flag", loadbalancingFlag)

      return message
    else:
      m = self.receivePattern.search(line)
      if m:
        toRank = m.group(1)
        loadbalancingFlag = m.group(2) 
        
        if loadbalancingFlag == "-2":
          loadbalancingFlag = "Continue"
        elif loadbalancingFlag == "-1":
          loadbalancingFlag = "Join"
        elif loadbalancingFlag == "0":
          loadbalancingFlag = "Undefined"
        elif loadbalancingFlag == "1":
          loadbalancingFlag = "ForkOnce"
        elif int(loadbalancingFlag) > 1:
          loadbalancingFlag = "ForkGreedy"
        
        message = Message("ReceivedLoadBalancingMessage")
        message.addAttribute("To", toRank)
        message.addAttribute("flag", loadbalancingFlag)
  
        return message        
      else:
        return None
    
    