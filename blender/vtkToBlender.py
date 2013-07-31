import pickle
import struct
import vtk

class TXTFileWriter:
  def __init__(self, outputFilename):
    self.outputFilename = outputFilename
    self.points = dict()
    self.cells = []
    
  def writeCell(self, q, vtkCell, rank):
    cell = Cell(q, vtkCell, self.points, rank)
    cell.handlePoints()
    self.cells.append(cell)
    
  def close(self):
    with open(self.outputFilename, "wb") as f:
      pickle.dump(len(self.points), f)
      for pointId in self.points:
        pickle.dump(pointId, f)
        pickle.dump(self.points[pointId].x, f)
        pickle.dump(self.points[pointId].y, f)
        pickle.dump(self.points[pointId].getValue(), f)

      pickle.dump(len(self.cells), f)
      for cell in self.cells:
        pickle.dump(cell.rank, f)
        for index in cell.vertexIndices:
          pickle.dump(index, f)
    print("    --done writing file " + self.outputFilename + "--")

class VTKFileReader:
  def __init__(self, filename, fileWriter, rank):
    self.filename = filename
    self.fileWriter = fileWriter
    self.rank = rank

  def convert(self):
    print("Converting file " + self.filename + "...")
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(self.filename)
    reader.SetReadAllScalars(True)
    reader.Update()

    grid = reader.GetOutput()
    numberOfCells = grid.GetNumberOfCells()
    cellData = grid.GetCellData()
    qs = cellData.GetScalars("q0")

    for cellId in xrange(numberOfCells):
      vtkCell = grid.GetCell(cellId)
      q = qs.GetTuple(cellId)

      self.fileWriter.writeCell(q, vtkCell, self.rank)
    
    print("    --done reading file " + self.filename + "--")

class Cell:
  def __init__(self, q, vtkCell, points, rank):
    self.q = q
    self.vtkCell = vtkCell
    self.vertexIndices = []
    self.points = points
    self.rank = rank

  def handlePoints(self):
    bounds = self.vtkCell.GetBounds()
    self.handlePoint(bounds[0], bounds[2], 0)
    self.handlePoint(bounds[1], bounds[2], 1)
    self.handlePoint(bounds[1], bounds[3], 3)
    self.handlePoint(bounds[0], bounds[3], 2)

  def handlePoint(self, x, y, index):
    id = self.vtkCell.GetPointIds().GetId(index)
    point = self.points.get(id, None)
    if point == None:
      point = Point(x, y)
      self.points[id] = point
    point.addValue(self.q)
    self.vertexIndices.append(id)

class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.q = 0
    self.counter = 0

  def addValue(self, q):
    self.counter = self.counter + 1
    self.q = self.q + q[0]

  def getValue(self):
    if self.counter == 0:
      raise "No value added for point " + str(x) + "," + str(y)
    return self.q / self.counter

def convertFiles(sequenceName, numberOfIterations, numberOfRanks):
  import glob
  import os
  
  for iteration in xrange(numberOfIterations):
    outputFilename = sequenceName.replace('__ITERATION__', str(iteration)).replace('__RANK__', '0').replace('.vtk', '.txt')
    fileWriter = TXTFileWriter(outputFilename)
  
    for rank in xrange(numberOfRanks):
      inputFilename = sequenceName.replace('__ITERATION__', str(iteration)).replace('__RANK__', str(rank))
      
      if len(glob.glob(inputFilename)) == 1:
        print "Processing", inputFilename, "->", outputFilename
        vtkFile = VTKFileReader(inputFilename, fileWriter, rank)
        vtkFile.convert()
        
    fileWriter.close()
      
def main():
  import sys
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--iterations", type=int, help="Number of iterations")
  parser.add_argument("-r", "--ranks", type=int, help="Number of ranks", default=0)
  parser.add_argument("sequenceName", help="Filename stump for vtks. __RANK__ is replaced by rank number. __ITERATION__ is replaced by iteration number.")
  args = parser.parse_args()
  
  print "Iterations =", args.iterations, ", Ranks =", args.ranks
  convertFiles(args.sequenceName, args.iterations, args.ranks)

if __name__ == "__main__":
  main()



