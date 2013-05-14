import re
from Parser import Parser
from Message import Message

class MasterWorkerCommunicationParser(Parser):
  
  def __init__(self):
    self.prepareSendToWorkerPattern =  re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToWorker\(...\)\s*in:.*level:(\d+).*verticesEnumerator\.getVertexPosition\(0\):(.*).*worker:(\d+)")
    self.prepareSendToMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::prepareSendToMaster\(...\)\s*in:.*level:(\d+).*verticesEnumerator\.getVertexPosition\(0\):([^ ]+)")
    self.receiveDataFromMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::receiveDataFromMaster\(...\)\s*in:")
    self.mergeWithWorkerPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithWorker\(...\)\s*in:")
    self.mergeWithMasterPattern = re.compile("rank:(\d+) peanoclaw::mappings::Remesh::mergeWithMaster\(...\)\s*in:.*worker:(\d+)")
    
  def parseLine(self, line):
    m = self.prepareSendToWorkerPattern.search(line)
    if m:
      fromRank = m.group(1)
      position = m.group(2)
      level = m.group(3)
      toRank = m.group(4)
      message = Message("PrepareSendToWorker")
      message.addAttribute("From", fromRank)
      message.addAttribute("To", toRank)
      message.addAttribute("Level", level)
      message.addAttribute("Position", position)
      return message
    else:
      m = self.prepareSendToMasterPattern.search(line)
      if m:
        fromRank = m.group(1)
        level = m.group(2)
        position = m.group(3)
        message = Message("PrepareSendToMaster")
        message.addAttribute("From", fromRank)
        message.addAttribute("Level", level)
        message.addAttribute("Position", position)
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
