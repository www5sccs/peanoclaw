import bpy
import sys
import pickle
import struct
import numpy

PEANO_PREFIX = "_Peano"
WATER_MATERIAL_NAME = "Meta-water"
WIREFRAME_MATERIAL = "WireframeMaterial"
WIREFRAME_OFFSET = 0.001

class ReferenceArray:
  def __init__(self, cellsPerDimension):
    numpy.zeros([cellsPerDimension, cellsPerDimension], int)

def deletePeanoObjects():
  r"""Unlinks all objects starting with PEANO_PREFIX
      from Blender."""
  for objekt in bpy.data.objects:
    if(objekt.name.startswith(PEANO_PREFIX)):
      bpy.context.scene.objects.unlink(objekt)
      bpy.data.objects.remove(objekt)
  for mesh in bpy.data.meshes:
    if(mesh.name.startswith(PEANO_PREFIX)):
      bpy.data.meshes.remove(mesh)
      

def importFile(filename, timestepNumber):

  #Deselect all objects
  for obj in bpy.data.objects:
    obj.select = False

  #Create object
  objectName = PEANO_PREFIX + "Object" + str(timestepNumber)
  coords=[]
  faces=[]
  with open(filename, "rb") as f:
    numberOfVertices = pickle.load(f)
    coords = [0] * numberOfVertices
    for vertexIndex in range(numberOfVertices):
      vertexId = pickle.load(f)
      x = pickle.load(f)
      y = pickle.load(f)
      q = pickle.load(f)
      coords[vertexId] = (x, y, q/10.0)

    numberOfCells = pickle.load(f)
    for cellId in range(numberOfCells):
      rank = pickle.load(f)
      vertexIds = []
      for i in range(4):
        vertexIds.append(pickle.load(f))
      faces.append((vertexIds[0], vertexIds[1], vertexIds[2], vertexIds[3]))
  
  mesh = bpy.data.meshes.new(PEANO_PREFIX + "Mesh" + str(timestepNumber))
  peanoObject = bpy.data.objects.new(objectName, mesh)
  peanoObject.location = (0, 0, 0)
  bpy.context.scene.objects.link(peanoObject)
  peanoObject.select = True
  bpy.context.scene.objects.active = peanoObject

  #Wireframe
  wireframeObject = bpy.data.objects.new(objectName + "Wireframe", mesh)
  wireframeObject.location = (0, 0, WIREFRAME_OFFSET)
  bpy.context.scene.objects.link(wireframeObject)
  wireframeObject.select = True

  #Mesh
  mesh.from_pydata(coords, [], faces)
  mesh.update(calc_edges=True)
  bpy.ops.object.mode_set(mode = 'EDIT')
  bpy.ops.mesh.select_all(action='SELECT')
  #bpy.ops.mesh.remove_doubles(limit=0.00001)
  mesh.calc_normals()
  mesh.auto_smooth_angle = 10.0
  mesh.use_auto_smooth = True
  bpy.ops.mesh.faces_shade_smooth()
  bpy.ops.object.editmode_toggle()
  bpy.ops.object.mode_set(mode = 'EDIT')
  bpy.ops.mesh.select_all(action='SELECT')
  bpy.ops.mesh.faces_shade_smooth()
  bpy.ops.object.editmode_toggle()

  #Set materials
  #Water material
  wireframeObject.select = False
  bpy.context.scene.objects.active = peanoObject
  material = bpy.data.materials[WATER_MATERIAL_NAME]
  bpy.ops.object.material_slot_add()
  peanoObject.material_slots[len(obj.material_slots) - 1].link = "OBJECT"
  peanoObject.material_slots[len(obj.material_slots) - 1].material = material
  bpy.ops.object.mode_set(mode = "EDIT")
  bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.object.material_slot_assign()
  bpy.ops.object.editmode_toggle()
  #Wireframe material
  wireframeObject.select = True
  peanoObject.select = False
  bpy.context.scene.objects.active = wireframeObject
  material = bpy.data.materials[WIREFRAME_MATERIAL]
  wireframeObject.material_slots[len(obj.material_slots) - 1].link = "OBJECT"
  wireframeObject.material_slots[len(obj.material_slots) - 1].material = material
  bpy.ops.object.mode_set(mode = "EDIT")
  bpy.ops.mesh.select_all(action="SELECT")
  bpy.ops.object.material_slot_assign()
  bpy.ops.object.editmode_toggle()
  wireframeObject.draw_type = "WIRE"
  
  #Decimate modifier
  #py.context.scene.objects.active = peanoObject
  #bpy.ops.object.modifier_add(type="DECIMATE")
  #peanoObject.modifiers[len(peanoObject.modifiers)-1].show_viewport = False
  #bpy.context.scene.objects.active = wireframeObject
  #bpy.ops.object.modifier_add(type="DECIMATE")
  #wireframeObject.modifiers[len(peanoObject.modifiers)-1].show_viewport = False

#  #Set key frames
#  peanoObject.select = True
#  wireframeObject.select = True
#  #t-1
#  bpy.context.scene.objects.active = peanoObject
#  bpy.ops.anim.change_frame(frame = timestepNumber-1)
#  peanoObject.scale = (0, 0, 0)
#  wireframeObject.scale = (0, 0, 0)
#  bpy.ops.anim.keyframe_insert_menu(type="Scaling")
#  peanoObject.location.z = 1.0e1
#  wireframeObject.location.z = 1.0e1
#  bpy.ops.anim.keyframe_insert_menu(type="Location")
#  #peanoObject.modifiers[len(peanoObject.modifiers)-1].ratio = 0.0
#  #peanoObject.modifiers[len(peanoObject.modifiers)-1].keyframe_insert(data_path="ratio")
#  #t
#  bpy.ops.anim.change_frame(frame = timestepNumber)
#  peanoObject.scale = (1, 1, 1)
#  wireframeObject.scale = (1, 1, 1)
#  bpy.ops.anim.keyframe_insert_menu(type="Scaling")  
#  peanoObject.location.z = 0
#  wireframeObject.location.z = WIREFRAME_OFFSET
#  bpy.ops.anim.keyframe_insert_menu(type="Location")  
#  #peanoObject.modifiers[len(peanoObject.modifiers)-1].ratio = 1.0
#  #peanoObject.modifiers[len(peanoObject.modifiers)-1].keyframe_insert(data_path="ratio")
#  #t+1
#  bpy.ops.anim.change_frame(frame = timestepNumber+1)
#  peanoObject.scale = (0, 0, 0)
#  wireframeObject.scale = (0, 0, 0)
#  bpy.ops.anim.keyframe_insert_menu(type="Scaling")
#  peanoObject.location.z = 1.0e1
#  wireframeObject.location.z = 1.0e1
#  bpy.ops.anim.keyframe_insert_menu(type="Location")  
#  peanoObject.select = False
#  wireframeObject.select = False
#  bpy.ops.anim.change_frame(frame = 0)

#def importFiles(sequenceName):
#  import glob
#  import os
#  import naturalSorting
#  import time
#  import sys

#  totalStartTime = time.clock()

#  print("Deleting old Peano objects...")
#  deletePeanoObjects()
#  print("    --done-- (%(time)fs)" % {"time": (time.clock() - totalStartTime)})

#  timestepNumber = 0
#  fileNames = glob.glob(sequenceName + "*.txt")
#  fileNames = naturalSorting.naturalSort(fileNames)
#  for filename in fileNames:
#    fileStartTime = time.clock()
#    print("Importing file " + filename)
#    importFile(filename, timestepNumber)
#    timestepNumber = timestepNumber + 1
#    print("    --done-- (%(time)fs)" % {"time": (time.clock() - fileStartTime)})
#    sys.stdout.flush()
#  print("    --Import completed-- (%(time)fs)" % {"time": (time.clock() - totalStartTime)})


def renderFiles(sequenceName, outputPath, numberOfIterations, cellsInReferenceArray):
  import glob
  import os
  import naturalSorting
  import time
  import sys

  totalStartTime = time.clock()

  timestepNumber = 0
  #fileNames = glob.glob(sequenceName + "*.txt")
  #fileNames = naturalSorting.naturalSort(fileNames)
  #for filename in fileNames:
  for iteration in xrange(numberOfIterations):
    fileName = sequenceName.replace("__ITERATION__", str(iteration))
    print("Deleting old Peano objects...")
    deletePeanoObjects()
    print("    --done-- (%(time)fs)" % {"time": (time.clock() - totalStartTime)})
  
    fileStartTime = time.clock()
    print("Importing file " + filename)
    importFile(filename, timestepNumber)
    bpy.ops.anim.change_frame(frame = timestepNumber)
    bpy.ops.render.render()
    bpy.data.images['Render Result'].save_render(filepath=outputPath+"%(timestep)05d.jpg" % {"timestep": timestepNumber})
    timestepNumber = timestepNumber + 1
    print("    --done-- (%(time)fs)" % {"time": (time.clock() - fileStartTime)})
    sys.stdout.flush()
  print("    --Import completed-- (%(time)fs)" % {"time": (time.clock() - totalStartTime)})


