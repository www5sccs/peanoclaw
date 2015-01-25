from vapory import *
import numpy

class RegularMesh2D(POVRayElement):

  def __init__(self, offsetX, offsetY, width, height, nx, ny):
    self.offsetX = offsetX
    self.offsetY = offsetY
    self.width = width
    self.height = height
    self.nx = nx
    self.ny = ny
    self.dx = float(width) / nx
    self.dy = float(height) / ny
    self.vertices = numpy.zeros((nx+1) * (ny+1))
    self.vertexContributions = numpy.zeros((nx+1) * (ny+1))
    self.faces = []
    self.vertexIndexMap = [0, 1, nx, nx + 1]
  
  def addCell(self, value, index):
    #print 'Cell', index
    for i in xrange(4):
      vertexIndex = self.mapToVertexIndex(index, i)
      #print ' Vertex', vertexIndex
      self.vertices[vertexIndex] += value
      self.vertexContributions[vertexIndex] += 1
    self.faces.append([self.mapToVertexIndex(index, 0), self.mapToVertexIndex(index, 3), self.mapToVertexIndex(index, 1)])
    self.faces.append([self.mapToVertexIndex(index, 0), self.mapToVertexIndex(index, 2), self.mapToVertexIndex(index, 3)])
    
  def finalizeInterpolation(self):
    for i in xrange(len(self.vertices)):
      if self.vertexContributions[i] > 0:
        self.vertices[i] /= self.vertexContributions[i]
      self.vertexContributions[i] = 0
    
  def mapToVertexIndex(self, cellIndex, localVertexIndex):
    row = cellIndex / self.nx
    if localVertexIndex == 0:
      return cellIndex + row
    elif localVertexIndex == 1:
      return cellIndex + row + 1
    elif localVertexIndex == 2:
      return cellIndex + row + self.nx + 1
    elif localVertexIndex == 3:
      return cellIndex + row + self.nx + 1 + 1
    else:
      raise Exception('Invalid localVertexIndex ' + str(localVertexIndex))

  def getVertexPosition(self, vertexIndex):
    x = vertexIndex % (self.nx+1)
    y = vertexIndex / (self.nx+1)
    return [self.offsetX + x * self.dx, 0.1 + 0.1 * self.vertices[vertexIndex], self.offsetY + y * self.dy]
    
  def __str__(self):
    meshEntries = ['mesh2 {\n']
    self.writeVertices(meshEntries)
    self.writeFaces(meshEntries)
    #meshEntries.append('  texture {WireframeHorizontalTexture} texture {WireframeHorizontalTexture rotate pi/2}')
    meshEntries.append('  texture {WireframeTexture scale <' + str(self.dx) + ',1,' + str(self.dy) + '>}')
    #meshEntries.append('  texture {pigment {color rgb <1,1,0,0.5>} finish{phong 1}}')
    meshEntries.append('}\n')
    return ''.join(meshEntries)
    
  def writeVertices(self, meshEntries):
    meshEntries.append('  vertex_vectors {\n')
    meshEntries.append('    ' + str(len(self.vertices)) + ', ')
    for i in xrange(len(self.vertices)):
      position = self.getVertexPosition(i)
      meshEntries.append('<' + ','.join(map(str, position)) + '> ')
    meshEntries.append('  }\n')
      
  def writeFaces(self, meshEntries):
    meshEntries.append('  face_indices {\n')
    meshEntries.append('    ' + str(len(self.faces)) + ', ')
    #meshEntries.append('    ' + str(25) + ', ')
    for face in self.faces:
      meshEntries.append('<' + ','.join(map(str, face)) + '> ')
    meshEntries.append('  }\n')  



