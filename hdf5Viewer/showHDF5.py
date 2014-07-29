import h5py
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file')
arguments = parser.parse_args()

f = h5py.File(arguments.file, 'r')

from mayavi import mlab

fig = mlab.figure()
#fig.scene.disable_render = True
#mlab.options.offscreen = True
for datasetName in f:
  print datasetName
  dataset = f[datasetName]

  position = dataset.attrs['position']
  size = dataset.attrs['size']

  x = []
  for i in xrange(dataset.shape[0]+1):
    x.append((position[0] + size[0] * (float(i) / dataset.shape[0])) * 10.0)
    
  y = []
  for i in xrange(dataset.shape[1]+1):
    y.append((position[1] + size[1] * (float(i) / dataset.shape[1])) * 10.0)
  
  
  mlab.surf(x, y, dataset)
  #source = mlab.pipeline.array2d_source(x, y, dataset)
  ##point_data = mlab.pipeline.cell_to_point_data(source)
  #point_data = mlab.pipeline.point_to_cell_data(source)
  #warp = mlab.pipeline.warp_scalar(point_data)
  ##surface = mlab.pipeline.surface(warp)
  #normals = mlab.pipeline.poly_data_normals(warp)
  #surface = mlab.pipeline.surface(warp)
  
#fig.scene.disable_render = False
mlab.show()


