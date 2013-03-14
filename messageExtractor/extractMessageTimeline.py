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

def parseFile(inputFile, outputFile):
  parsers = (HeapDataParser(), 
             LoadBalancingMessageParser(), 
             StateMessageParser(), 
             MasterWorkerCommunicationParser(), 
             ForkJoinCommunicationParser(), 
             NeighbourCommunicationParser())
  
  rootMessage = RootMessage()
  
  while True:
    line = inputFile.readline()
    if not line:
      break
  
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
  
  while True:
    numberOfRanks = len(glob.glob("it-" + "%05d" % iteration + "*.txt"))
    if numberOfRanks == 0:
      break
    
    htmlFile.write("<h1>Iteration " + str(iteration) + "</h1>\n")
    
    rank = 0
    print "------"
    while True:
      inputFileName = "it-" + "%05d" % iteration + "-rank-" + str(rank) + "-trace.txt"
      try:
        inputFile = open(inputFileName, 'r')
      except IOError:
        break
      print inputFileName
      outputFileName = inputFileName.replace(".txt", ".xml")
      outputFile = open(outputFileName, 'w')
      parseFile(inputFile, outputFile)
      
      style = 'style="width:' + str(100/numberOfRanks - 3) + '%; height:75%;"'
      htmlFile.write('<iframe src="'+ outputFileName + '" ' + style + '></iframe>')
      rank += 1
      
    htmlFile.write("\n")
    iteration += 1
  
  htmlFile.write("</body></html>\n")
