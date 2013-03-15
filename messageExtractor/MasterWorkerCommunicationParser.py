import re
from Parser import Parser
from Message import Message

class MasterWorkerCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareSendToWorkerPattern =  re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToWorker\(...\)\s*in:.*worker:(\d+)")
    self.prepareSendToMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToMaster\(...\)\s*in:")
    self.receiveDataFromMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::receiveDataFromMaster\(...\)\s*in:")
    self.mergeWithWorkerPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithWorker\(...\)\s*in:")
    self.mergeWithMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithMaster\(...\)\s*in:.*worker:(\d+)")
    
  def parseLine(self, line):
    m = self.prepareSendToWorkerPattern.search(line)
    if m:
      fromRank = m.group(1)
      toRank = m.group(2)
      message = Message("PrepareSendToWorker")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      return message
    else:
      m = self.prepareSendToMasterPattern.search(line)
      if m:
        fromRank = m.group(1)
        message = Message("PrepareSendToMaster")
        message.addAttribute("From", fromRank)
        return message
      else:
        m = self.receiveDataFromMasterPattern.search(line)
        if m:
          fromRank = m.group(1)
          message = Message("ReceiveDataFromMaster")
          message.addAttribute("From", fromRank)
          return message  
        else:
          m = self.mergeWithWorkerPattern.search(line)
          if m:
            fromRank = m.group(1)
            message = Message("MergeWithWorker")
            message.addAttribute("From", fromRank)
            return message            
          else:
            m = self.mergeWithMasterPattern.search(line)
            if m:
              fromRank = m.group(1)
              message = Message("MergeWithMaster")
              message.addAttribute("From", fromRank)
              return message
            else:
              return None
