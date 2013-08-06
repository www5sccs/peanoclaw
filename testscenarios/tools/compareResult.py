accuracy = 1e-8

class Cell:
  def __init__(self, vtkCell, bounds, q):
    self.vtkCell = vtkCell
    self.bounds = bounds
    self.q = q
    
  def __eq__(self, other):
    global accuracy
    if abs(self.q - other.q) > accuracy:
      return false
    
    if len(sel.bounds) != len(other.bounds):
      return false
    
    for i in xrange(len(self.bounds)):
      if abs(self.bounds[i] - other.bounds[i]) > accuracy:
        return false
      
    return true
  
  def __cmp__(self, other):
    global accuracy
    if self.q - other.q > accuracy:
      return 1
    elif other.q - self.q > accuracy:
      return -1
    
    if len(self.bounds) != len(other.bounds):
      return false
    
    for i in xrange(len(self.bounds)):
      if self.bounds[i] - other.bounds[i] > accuracy:
        return 1
      elif other.bounds[i] - self.bounds[i] > accuracy:
        return -1
    
    return 0
  
  def __str__(self):
    return "q: " + str(self.q) + " bounds: " + str(self.bounds)

def parseRange(argument):
  if ':' in argument:
    return range(*map(int, argument.split(':')))
  return range(int(argument), int(argument)+1)

def readCellsFromFile(cells, path, iteration, rank):
  import vtk
  filename = path.replace('__ITERATION__', str(iteration)).replace('__RANK__', str(rank))

  reader = vtk.vtkDataSetReader()
  reader.SetFileName(filename)
  reader.SetReadAllScalars(True)
  reader.Update()

  grid = reader.GetOutput()
  numberOfCells = grid.GetNumberOfCells()
  cellData = grid.GetCellData()
  qs = cellData.GetScalars("q0")

  for cellId in xrange(numberOfCells):
    vtkCell = grid.GetCell(cellId)
    
    q = qs.GetTuple(cellId)[0]
    cells.append(Cell(vtkCell, vtkCell.GetBounds()[:], q))
    
  return numberOfCells

def findClosestMatch(cell, cells):
  bestIndex = -1
  minDistance = 1000000
  import math
  for index in xrange(len(cells)):
    c = cells[index]
    distance = 0
    for i in xrange(len(cell.bounds)):
      distance += (cell.bounds[i] - c.bounds[i])**2
    distance = math.sqrt((c.q - cell.q)**2 * 10 + distance)
    
    if distance < minDistance:
      minDistance = distance
      bestIndex = index
  return bestIndex

def findCellInList(cell, cells):
  lower = 0
  upper = len(cells)
  
  while(upper > lower):
    middle = (upper + lower) / 2
    middleCell = cells[middle]
    if middleCell < cell:
      lower = middle + 1
    elif middleCell > cell:
      upper = middle
    else:
      return middle
  
  return -1

def main():
  from argparse import ArgumentParser
  parser = ArgumentParser(description='Tool for comparing vtk output of parallel runs.')
  parser.add_argument('path1', help='The path to the first set of vtk files.')
  parser.add_argument('path2', help='The path to the second set of vtk files.')
  parser.add_argument('iteration1', type=int, help='The iteration number of the first set of vtk files.')
  parser.add_argument('ranks1', help='The range of ranks for the first set of vtk files.')
  parser.add_argument('iteration2', type=int, help='The iteration number of the second set of vtk files.')
  parser.add_argument('ranks2', help='The range of ranks for the second set of vtk files.')
  parser.add_argument('accuracy', help='The range of ranks for the second set of vtk files.', type=float, nargs='?', const='1e-5')
  arguments = parser.parse_args()
  
  global accuracy
  accuracy = arguments.accuracy
  if arguments.path2 == 'SameAsPath1':
    path2 = arguments.path1
  else:
    path2 = arguments.path2
  
  #Loop through ranks1
  cells1 = [] #set()
  ranks1 = parseRange(arguments.ranks1)
  for rank in ranks1:
    print "1: Parsing rank...", rank
    numberOfCells = readCellsFromFile(cells1, arguments.path1, arguments.iteration1, rank)
    print "Read", numberOfCells, "cells."
  print "1: Total number of cells:", len(cells1)
    
  #Loop through ranks2
  cells2 = [] #set()
  ranks2 = parseRange(arguments.ranks2)
  
  print ranks2
  
  for rank in ranks2:
    print "2: Parsing rank", rank
    numberOfCells = readCellsFromFile(cells2, path2, arguments.iteration2, rank)
    print "Read", numberOfCells, "cells."
  print "2: Total number of cells:", len(cells2)
    
  #Compare lists
  if len(cells1) != len(cells2):
    raise Exception("Number of cells do not match!")
    
  cells1.sort()
  cells2.sort()
  
  for cell in cells1:
    index = findCellInList(cell, cells2)
    
    if index == -1:
      bestMatch = findClosestMatch(cell, cells2)
      if bestMatch == -1:
        bestMatchString = ""
      else:
        bestMatchString = "Best match is " + str(cells2[bestMatch])
      raise Exception("No matching cell for " + str(cell) + ". " + bestMatchString)
    else:
      del cells2[index]
   
  print "All cells match" 

if __name__=="__main__":
  main()