
class Attribute(object):
  def __init__(self, name, value):
    self.name = name
    self.value = value

class Message(object):
  
  def __init__(self, messageType):
    self.number = 0
    self.attributes = []
    self.addAttribute("messageType", messageType)
    self.childMessages = []
    
  def addAttribute(self, name, value):
    self.attributes.append(Attribute(name, value))
    
  def getAttributes(self):
    return self.attributes
  
  def getName(self):
    return self.getAttributes()[0].value
  
  def isSameOrChildMessage(self, message):
    if len(message.getAttributes()) < len(self.attributes):
      return False
    
    for i in xrange(len(self.attributes)):
      if message.getAttributes()[i].name != self.getAttributes()[i].name:
        return False
      
      if message.getAttributes()[i].value != self.getAttributes()[i].value:
        return False
      
    return True
  
  def addRecursively(self, message):
    """
    Adds a new child to this method and calls insert method recursively.
    """
    newChild = Message(message.getName())
    
    for i in xrange(1, len(self.attributes) + 1):
      newChild.addAttribute(message.getAttributes()[i].name, message.getAttributes()[i].value)
      
    self.childMessages.append(newChild)
    newChild.insertMessage(message)
    
  def insertMessage(self, message):
    """
    Inserts the given message into this message or recursively
    into a child message.
    """
    if not self.isSameOrChildMessage(message):
      #print self.getName() + ": Rejected (Not same or child)"
      return False
    
    #for childMessage in self.childMessages:
    #  if childMessage.insertMessage(message):
    #    #print self.getName() + ": Added in subtree"
    #    self.number += 1
    #    return True
    if len(self.childMessages) > 0:
      if self.childMessages[-1].insertMessage(message):
        #print self.getName() + ": Added in subtree"
        self.number += 1
        return True

    if len(message.getAttributes()) == len(self.getAttributes()):
      pass
      #print self.getName() + ": Added as same"
    elif len(message.getAttributes()) == len(self.getAttributes()) + 1:
      #print self.getName() + ": Added as child"
      message.number = 1
      self.childMessages.append(message)
    else:
      #print self.getName() + ": Added recursively"
      self.addRecursively(message)
      
    self.number += 1
    return True

  def printXML(self, indent, file):
    """
    Prints this message in XML style.
    """
    startTag = "<" + self.getName() +  " "
    
    if self.number > 1:
      startTag += "number=" + "\"" + str(self.number) + "\" "
    
    for attribute in self.getAttributes():
      if attribute.name != "messageType":
        startTag += attribute.name + "=\"" + attribute.value + "\" "
    
    if len(self.childMessages) == 0:
      startTag += "/>"
    else:
      startTag += ">"
      
    file.write(("  " * indent) + startTag + "\n")
    
    #Descend
    for childMessage in self.childMessages:
      childMessage.printXML(indent + 1, file)
      
    if len(self.childMessages) > 0:
      endTag = "</" + self.getName() + ">"
      file.write(("  " * indent) + endTag + "\n")

class RootMessage(Message):
  
  def __init__(self):
    super(RootMessage, self).__init__("Root")
    self.attributes = []

  def getName(self):
    return "Root"
  
  def printXML(self, file):
    super(RootMessage,self).printXML(0, file)