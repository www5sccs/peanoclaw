import re
from Parser import Parser
from Message import Message

class MasterWorkerCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareSendToWorkerPattern =  "rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToWorker\(...\).*worker:(\d+)"
    self.prepareSendToMasterPattern = "rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToMaster\(...\)"
    self.receiveDataFromMasterPattern = "rank:(\d+) peanoclaw::mappings::SolveTimestep::receiveDataFromMaster\(...\)"
    self.mergeWithWorkerPattern = "rank:(\d+) peanoclaw::mappings::SolveTimestep::mergeWithWorker\(...\)"
    self.mergeWithMasterPattern = "rank:(\d+) peanoclaw::mappings::Remesh::mergeWithMaster\(...\).*worker:(\d+)"
    
  def parseLine(self, line):
    m = re.search(self.prepareSendToWorkerPattern, line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(2)
      message = Message("PrepareSendToWorker")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      return message
    else:
      m = re.search(self.prepareSendToMasterPattern, line)
      if m:
        fromRank = m.group(1)
        message = Message("PrepareSendToMaster")
        message.addAttribute("From", fromRank)
        return message
      else:
        m = re.search(self.receiveDataFromMasterPattern, line)
        if m:
          fromRank = m.group(1)
          message = Message("ReceiveDataFromMaster")
          message.addAttribute("From", fromRank)
          return message  
        else:
          m = re.search(self.mergeWithWorkerPattern, line)
          if m:
            fromRank = m.group(1)
            message = Message("MergeWithWorker")
            message.addAttribute("From", fromRank)
            return message            
          else:
            m = re.search(self.mergeWithMasterPattern, line)
            if m:
              fromRank = m.group(1)
              message = Message("MergeWithMaster")
              message.addAttribute("From", fromRank)
              return message
            else:
              return None
