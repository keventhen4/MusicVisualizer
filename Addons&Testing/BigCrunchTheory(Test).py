from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QQuaternion
from random import randint, uniform
import sys

app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
#w.showMaximized()
w.setGeometry(0, 110, 960, 540)
w.setWindowTitle('pyqtgraph example: GLMeshItem')
w.setCameraPosition(distance=40)
w.show()


#g = gl.GLGridItem()
#g.scale(2,2,1)
#w.addItem(g)

verts = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [1, 2, 0],
    [1, 1, 1],
])
faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
])
colors = np.array([
    [1, 0, 0, 0.3],
    [0, 1, 0, 0.3],
    [0, 0, 1, 0.3],
    [1, 1, 0, 0.3]
])


md = gl.MeshData.sphere(rows=4, cols=4)

colors = np.ones((md.faceCount(), 4), dtype=float)
colors[::2,0] = 0
colors[:,1] = np.linspace(0, 1, colors.shape[0])
md.setFaceColors(colors)
m3 = gl.GLMeshItem(meshdata=md, smooth=False)#, shader='balloon')
w.addItem(m3)


meshCount = 0
meshNumList = []
meshDataDict = {}
meshItemDict = {}
meshOffsetDict = {}
meshPosDict = {}
meshColorDict = {}

def RandomUniform(start, end): #Both Inclusive
    randomUniform = 0
    while randomUniform == 0:
        randomUniform = uniform(start, end)
    return randomUniform

while(1):
    q = QQuaternion(1, 0, 0, 1).normalized()
    axis, angle = q.getAxisAndAngle()
    m3.rotate(0.5, axis.x(), axis.y(), axis.z())
    
    if randint(0,1) == 0: #Object spawning
        meshCount += 1
        meshDataDict[f"mesh{meshCount}"] = gl.MeshData.sphere(rows=4,cols=4,radius=randint(1,10))
        meshItemDict[f"mesh{meshCount}"] = gl.GLMeshItem(meshdata = meshDataDict[f"mesh{meshCount}"], drawFaces = True, drawEdges = False, smooth = True)#, color=(1, 0, 0, 0.1))
        meshColorDict[f"mesh{meshCount}"] = [255, 255, 255, 1]
        meshOffsetDict[f"mesh{meshCount}"] = [RandomUniform(-0.1,0.1),RandomUniform(-0.1,0.1),RandomUniform(-0.1,0.1)]
        meshPosDict[f"mesh{meshCount}"] = [0, 0, 0]
        w.addItem(meshItemDict[f"mesh{meshCount}"])

    for meshNum in range(1,meshCount+1): #Movement and position update for out-of-bounds detection
        meshItemDict[f"mesh{meshNum}"].translate(meshOffsetDict[f"mesh{meshNum}"][0],meshOffsetDict[f"mesh{meshNum}"][1],meshOffsetDict[f"mesh{meshNum}"][2])
        meshPosDict[f"mesh{meshNum}"][0] += meshOffsetDict[f"mesh{meshNum}"][0]
        meshPosDict[f"mesh{meshNum}"][1] += meshOffsetDict[f"mesh{meshNum}"][1]
        meshPosDict[f"mesh{meshNum}"][2] += meshOffsetDict[f"mesh{meshNum}"][2]
    
    removedPos = []
    for meshNum in range(1, meshCount+1): #Detection + deletion from screen
        for axis in meshPosDict[f"mesh{meshNum}"]:
            if meshColorDict[f"mesh{meshNum}"][3] <= 0:
                w.removeItem(meshItemDict[f"mesh{meshNum}"]) #Removes object from screen
                currentRemovedLen = len(removedPos)
                removedPos.append(meshNum-currentRemovedLen) #Add mesh number to list of removed meshes
                break
            elif axis > 50 or axis < -50:
                meshColorDict[f"mesh{meshNum}"][3] -= 0.1
                meshItemDict[f"mesh{meshNum}"].setColor(c=(255, 255, 255, meshColorDict[f"mesh{meshNum}"][3]))
                break
            


    if len(removedPos) > 0: #Deletion from lists
        for position in removedPos:
            for meshNum in range(position, meshCount):
                meshDataDict[f"mesh{meshNum}"] = meshDataDict.pop(f"mesh{meshNum+1}")
                meshItemDict[f"mesh{meshNum}"] = meshItemDict.pop(f"mesh{meshNum+1}")
                meshOffsetDict[f"mesh{meshNum}"] = meshOffsetDict.pop(f"mesh{meshNum+1}")
                meshPosDict[f"mesh{meshNum}"] = meshPosDict.pop(f"mesh{meshNum+1}")
            meshCount -= 1

    app.processEvents()



## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()