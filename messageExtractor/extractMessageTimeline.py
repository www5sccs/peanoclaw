#!/usr/bin/python
import sys
import re
from Message import Message
from Message import RootMessage

from HeapDataParser import HeapDataParser
from LoadBalancingMessageParser import LoadBalancingMessageParser
from StateMessageParser import StateMessageParser
from MasterWorkerCommunicationParser import MasterWorkerCommunicationParser
from ForkJoinCommunicationParser import ForkJoinCommunicationParser
from NeighbourCommunicationParser import NeighbourCommunicationParser
from reportlab.lib.testutils import outputfile
  
class ParseTask(object):
  
  def __init__(self, fileName, inputFile, outputFile):
    self.fileName = fileName
    self.inputFile = inputFile
    self.outputFile = outputFile
    
  def run(self):
    self.parseFile(self.inputFile, self.outputFile)
    print("Finished " + self.fileName)
    
    
def parseFile(inputFile, outputFile):
  parsers = (HeapDataParser(),
             LoadBalancingMessageParser(), 
             StateMessageParser(), 
             MasterWorkerCommunicationParser(), 
             ForkJoinCommunicationParser(), 
             NeighbourCommunicationParser())
  
  rootMessage = RootMessage()
  
  excludePatterns = (re.compile("tarch::parallel::NodePool::replyToRegistrationMessages"), 
                     re.compile("tarch::parallel::NodePool::replyToWorkerRequestMessages"))
  
  #while True:
  #  line = inputFile.readline()
  #  if not line:
  #    break
  
  lines = inputFile.readlines()
  for line in lines:
    
    #skip = False
    #for excludePattern in excludePatterns:
    #  if excludePattern.search(line):
    #    skip = True
    #    break
    
    for parser in parsers:
      message = parser.parseLine(line)
      if message != None:
        rootMessage.insertMessage(message)
  
  rootMessage.printXML(outputFile)
      
      
if __name__ == "__main__":
  import sys
  import glob
  
  iteration = 1
  
  htmlFile = open("messages.html", 'w')
  htmlFile.write("<html><body>\n")
  
  #numberOfRanks = len(glob.glob("it-" + "%05d" % iteration + "*.txt"))
  if len(sys.argv) < 2:
    raise Exception("Usage: extractMessageTimeline <numberOfRanks>")
  numberOfRanks = int(sys.argv[1])
  
  from threadpool import ThreadPool
  threadpool = ThreadPool(3)
  
  while True:
    if len(glob.glob("it-" + "%05d" % iteration + "*.txt")) == 0:
      break
    
    htmlFile.write("<h1>Iteration " + str(iteration) + "</h1>\n")
    
    print "------"
    for rank in xrange(0, numberOfRanks):
      inputFileName = "it-" + "%05d" % iteration + "-rank-" + str(rank) + "-trace.txt"
      #style = 'style="width:' + str(100/numberOfRanks - 3) + '%; height:75%;"'
      style = 'style="width: 400px; height:75%;"'
      try:
        inputFile = open(inputFileName, 'r')
      except IOError:
        print "No file for rank " + str(rank) + " in iteration " + str(iteration)
        htmlFile.write('<iframe src="../nomessages.html" ' + style + '></iframe>')
        continue
      outputFileName = inputFileName.replace(".txt", ".xml")
      print inputFileName + " -> " + outputFileName
      outputFile = open(outputFileName, 'w')
      
      parseFile(inputFile, outputFile)
      
      #threadpool.add_task(ParseTask(outputFileName, inputFile, outputFile))
      
      htmlFile.write('<iframe src="'+ outputFileName + '" ' + style + '></iframe>')
      
    htmlFile.write("\n")
    htmlFile.flush()
    iteration += 1
    
    #if iteration >= 4:
    #  break
  
  #threadpool.wait_completion()
  
  htmlFile.write("</body></html>\n")
