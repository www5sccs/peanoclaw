from vapory import *
import h5py
import numpy
from os.path import exists
from mesh import RegularMesh2D

def createMeshObjects(meshFile):
  meshObjects = []

  hdf5File = h5py.File(meshFile)
  for datasetName in hdf5File:
    dataset = hdf5File[datasetName]
    
    position = dataset.attrs['position']
    size = dataset.attrs['size']
    
    meshObject = RegularMesh2D(position[0], position[1], size[0], size[1], dataset.shape[0], dataset.shape[1])
    
      #dimension = Dimension(str(d), position[d], position[d] + size[d], dataset.shape[d])
      #dimensions.append(dimension)
      
    iterator = numpy.nditer(dataset, flags=['multi_index', 'c_index'])
    while not iterator.finished:
      #print iterator.multi_index, iterator.index
      meshObject.addCell(iterator[0], iterator.index)
      iterator.iternext()
    meshObject.finalizeInterpolation()
      
    meshObjects.append(meshObject)
  return meshObjects


def render(meshFile, outputName):
  sun = LightSource([1000,2500,-2500], 'color', 'White')
  sky = Sphere( [0,0,0],1, 'hollow',
                Texture( Pigment( 'gradient', [0,1,0],
                         ColorMap([0.0, 'color', 'White'],
                                  [0.5, 'color', 'CadetBlue'],
                                  [1.0, 'color', 'CadetBlue']),
                         "quick_color", "White"),
                         Finish( 'ambient', 1, 'diffuse', 0)
              ), 'scale', 10000)
  ground = Plane( [0,1,0], 0,
                  Texture( Pigment( 'color', [0.85,0.55,0.30]), Finish( 'phong', 0.1) )
           )
  objects = [sun, sky, ground]
  
  if exists(meshFile):
    objects.extend(createMeshObjects(meshFile))
  
  scene = Scene( Camera('angle', 75,
                        'location', [1.5, 1.5, 0.5],
                        'look_at', [0.5 , 0.25 , 0.5]),
               objects = objects,
               included = ["colors.inc", "textures.inc", "glass.inc", 'wireframe.inc'],
               defaults = [Finish( 'ambient', 0.1, 'diffuse', 0.9)] )   
  scene.render(outputName, antialiasing=0.1, width=1024, height=768, remove_temp=False)

