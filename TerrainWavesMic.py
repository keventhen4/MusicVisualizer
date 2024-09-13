
import numpy as np
from random import randint, uniform
from opensimplex import OpenSimplex
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore , QtGui
#PyQt5.QtWidgets used instead of QtGui because QtGui doesn't have attribute QApplication
from PyQt5 import QtWidgets
#PyQt5.QtGui contains quaternion manipulation and conversion :(
from PyQt5.QtGui import QQuaternion
import struct
import pyaudio, aubio
import sys

def RandomUniform(start, end): #Both Inclusive
    randomUniform = 0
    while randomUniform == 0:
        randomUniform = uniform(start, end)
    return randomUniform

class Terrain(object):
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setGeometry(0, 110, 960, 540)
        self.window.show()
        self.window.setWindowTitle("Terrain")
        self.window.setCameraPosition(distance=40, elevation=0) #40,0 for side, 100,50 for far view
        self.window.show()
        #self.window.setBackgroundColor((0, 0, 0)) #sets the background of screen and somehow changes some colors?
        
        #The grid is the original grid first displayed as a base

        #grid = gl.GLGridItem()
        #grid.scale(2, 2, 2)
        #self.window.addItem(grid)
        self.nsteps = 1.3

        #self.spoint1 = np.array([0, -5, 0])
        #self.spoint2 = np.array([0, 5, 0])
        #self.scenter = (self.spoint1 + self.spoint2) / 2
        #self.sradius = np.linalg.norm(self.spoint2 - self.spoint1) / 2
        #self.sradius = 1.1

        self.ypoints = np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.xpoints = np.arange(-20, 20 + self.nsteps, self.nsteps)
        self.nfaces = len(self.ypoints)
        self.offset = 0

        self.smeshCount = 0
        self.smeshNumList = []
        self.smeshDataDict = {}
        self.smeshItemDict = {}
        self.smeshOffsetDict = {}
        self.smeshPosDict = {}
        self.smeshColorDict = {}

        self.smeshSpawnVolume = 15
        self.smeshSpawnDelayO = 250
        self.smeshSpawnDelay = self.smeshSpawnDelayO
        self.smeshSpawnTimeO = 5
        self.smeshSpawnTime = self.smeshSpawnTimeO

        self.RATE = 44100
        self.CHUNK = len(self.xpoints) * len(self.ypoints)

        self.PPyAudio = pyaudio.PyAudio()
        self.stream = self.PPyAudio.open(
            format = pyaudio.paInt16,
            channels = 1,
            rate = self.RATE,
            input = True,
            output = True,
            frames_per_buffer = self.CHUNK,
        )
        
        self.volumeBUFFER_SIZE = 2048
        self.volumeCHANNELS = 1
        self.volumeFORMAT = pyaudio.paFloat32
        self.volumeMETHOD = "default"
        self.volumeSAMPLE_RATE = 44100
        self.volumeHOP_SIZE = self.volumeBUFFER_SIZE//2
        self.volumePERIOD_SIZE_IN_FRAME = self.volumeHOP_SIZE
        
        self.volumeStream = self.PPyAudio.open(
            format=self.volumeFORMAT,
            channels=self.volumeCHANNELS,
            rate=self.volumeSAMPLE_RATE,
            input=True,
            frames_per_buffer=self.volumePERIOD_SIZE_IN_FRAME)

        self.noise = OpenSimplex(seed=0)

        verts, faces, colors = self.mesh()

        self.mesh1 = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors,
            smooth=False, #False for sharper color gradients, True for calmer and interesting but pale gradients
            drawEdges=True, #Will display a white line when in side view if True
        )
        self.mesh1.setGLOptions('additive')
        self.window.addItem(self.mesh1)
    

    def mesh(self, offset=0, height=0, audio_data=None):

        if audio_data is not None:
            audio_data = struct.unpack(str(2 * self.CHUNK) + 'B', audio_data)

            #np.array(value).astype(dtype) used instead as out-of-bound numbers will not be supported by numpy in future

            #audio_data = np.array(audio_data, dtype='b')[::2] + 128
            #audio_data = np.array(audio_data, dtype='int32') - 128
            audio_data = np.array(audio_data).astype(dtype='b')[::2] + 128
            audio_data = np.array(audio_data).astype(dtype='int32') - 128
            audio_data = audio_data * 0.02 #Multiplier to decrease height
            audio_data = audio_data.reshape((len(self.xpoints), len(self.ypoints)))
        else:
            audio_data = np.array([1] * 1024)
            audio_data = audio_data.reshape((len(self.xpoints), len(self.ypoints)))

        faces = []
        colors= []
        verts = np.array([
            [
                x, y, audio_data[xid][yid] * self.noise.noise2(x = xid / 200 + offset, y = yid / 200 + offset) * 2 #xid and yid division by 5, multiply by 4 for side view
            ] for xid, x in enumerate(self.xpoints) for yid, y in enumerate(self.ypoints)
        ], dtype=np.float32)
        for yid in range(self.nfaces - 1):
            yoffset = yid * self.nfaces
            for xid in range(self.nfaces - 1):
                faces.append([
                    xid + yoffset,
                    xid + yoffset + self.nfaces,
                    xid + yoffset + self.nfaces + 1,
                ])
                faces.append([
                    xid + yoffset,
                    xid + yoffset + 1,
                    xid + yoffset + self.nfaces + 1,
                ])
                #colors append appends rgb values (probably) and a 4th value opacity
                #original rainbow colors:
                # xid / self.nfaces, 1 - xid / self.nfaces, yid / self.nfaces, 1
                colors.append([
                    xid / self.nfaces, 5 - xid / self.nfaces, yid / self.nfaces, 0.75 #xid / self.nfaces, 5 - xid / self.nfaces, yid / self.nfaces, 1
                ])
                colors.append([
                    xid / self.nfaces, 5 - xid / self.nfaces, yid / self.nfaces, 0.75 #xid / self.nfaces, 5 - xid / self.nfaces, yid / self.nfaces, 1
                ])
        
        faces = np.array(faces).astype(dtype=np.uint32)
        colors = np.array(colors).astype(dtype=np.float32)

        return verts, faces, colors


    def smeshdetection(self):
        removedPos = []
        for meshNum in range(1, self.smeshCount+1): #Detection + deletion from screen
            for axis in self.smeshPosDict[f"mesh{meshNum}"]:
                if axis > 21 or axis < -21:
                    if 1 < len(self.smeshItemDict) < 4:
                        for meshNum1 in range(1,len(self.smeshItemDict)+1):
                            self.window.removeItem(self.smeshItemDict[f"mesh{meshNum1}"]) #Remove objects if more 3 or less
                            currentRemovedLen = len(removedPos)
                            removedPos.append(meshNum1-currentRemovedLen)
                        return removedPos
                    else:
                        self.window.removeItem(self.smeshItemDict[f"mesh{meshNum}"]) #Removes object from screen
                        currentRemovedLen = len(removedPos)
                        removedPos.append(meshNum-currentRemovedLen) #Add mesh number to list of removed meshes
                        break
                elif axis > 20 or axis < -20:
                    self.smeshColorDict[f"mesh{meshNum}"][0] -= 10 if self.smeshColorDict[f"mesh{meshNum}"][0] > 10 else 0
                    self.smeshColorDict[f"mesh{meshNum}"][1] -= 10 if self.smeshColorDict[f"mesh{meshNum}"][1] > 10 else 0
                    self.smeshColorDict[f"mesh{meshNum}"][2] -= 10 if self.smeshColorDict[f"mesh{meshNum}"][2] > 10 else 0
                    #self.smeshItemDict[f"mesh{meshNum}"].setMeshData(color=(255, 255, 255, ))
                    #meshItemDict[f"mesh{meshNum}"].setMeshData(colors=(1,1,1,meshColorDict[f"mesh{meshNum}"][3]))
                
                    self.smeshItemDict[f"mesh{meshNum}"].setColor(
                        QtGui.QColor(
                            self.smeshColorDict[f"mesh{meshNum}"][0],
                            self.smeshColorDict[f"mesh{meshNum}"][1],
                            self.smeshColorDict[f"mesh{meshNum}"][2]
                        )
                    )
                    break
        return removedPos


    def smeshdeletion(self, removedPos):
        if len(removedPos) > 0: #Deletion from lists
            for position in removedPos:
                for meshNum in range(position, self.smeshCount):
                    self.smeshDataDict[f"mesh{meshNum}"] = self.smeshDataDict.pop(f"mesh{meshNum+1}")
                    self.smeshItemDict[f"mesh{meshNum}"] = self.smeshItemDict.pop(f"mesh{meshNum+1}")
                    self.smeshOffsetDict[f"mesh{meshNum}"] = self.smeshOffsetDict.pop(f"mesh{meshNum+1}")
                    self.smeshPosDict[f"mesh{meshNum}"] = self.smeshPosDict.pop(f"mesh{meshNum+1}")
                self.smeshCount -= 1
        removedPos = []


    def smesh(self, volume): #spheres
        
        if self.smeshSpawnDelay > 0:
            self.smeshSpawnDelay -= 1
        elif int(float(volume)*1000000) < self.smeshSpawnVolume and self.smeshSpawnTime == 0 and self.smeshSpawnDelay == 0:
            self.smeshSpawnTime = self.smeshSpawnTimeO
            self.smeshCount += 1
            #Starting data and size, then end size for increases
            self.smeshDataDict[f"mesh{self.smeshCount}"] = gl.MeshData.sphere(rows=10,cols=20,radius=0.1)
            self.smeshOffsetDict[f"mesh{self.smeshCount}"] = [
                RandomUniform(-0.1,0.1),
                RandomUniform(-0.1,0.1),
                RandomUniform(-0.1,0.1)]
            self.smeshPosDict[f"mesh{self.smeshCount}"] = [
                RandomUniform(-15,15),
                RandomUniform(-15,15),
                0]
            self.smeshColorDict[f"mesh{self.smeshCount}"] = [100, 100, 100, RandomUniform(0.01, 0.3)]
            #Goofy mesh item itself with funny options
            self.smeshItemDict[f"mesh{self.smeshCount}"] = gl.GLMeshItem(
                meshdata = self.smeshDataDict[f"mesh{self.smeshCount}"],
                color=(255, 255, 255, self.smeshColorDict[f"mesh{self.smeshCount}"][3]),
                smooth = True,
                shader="balloon",
                glOptions="additive",
                #drawFaces = True,
                #drawEdges = False,
            )
            self.smeshItemDict[f"mesh{self.smeshCount}"].translate(
                self.smeshPosDict[f"mesh{self.smeshCount}"][0],
                self.smeshPosDict[f"mesh{self.smeshCount}"][1],
                self.smeshPosDict[f"mesh{self.smeshCount}"][2])
            self.window.addItem(self.smeshItemDict[f"mesh{self.smeshCount}"])
        elif self.smeshSpawnDelay == 0:
            self.smeshSpawnTime -= 1

        if int(float(volume)*1000000) > self.smeshSpawnVolume:
            self.smeshSpawnDelay = self.smeshSpawnDelayO

        for smeshNum in range(1, self.smeshCount+1): #Movement and position update for out-of-bounds detection
            self.smeshItemDict[f"mesh{smeshNum}"].translate(
                self.smeshOffsetDict[f"mesh{smeshNum}"][0],
                self.smeshOffsetDict[f"mesh{smeshNum}"][1],
                self.smeshOffsetDict[f"mesh{smeshNum}"][2])
            self.smeshPosDict[f"mesh{smeshNum}"][0] += self.smeshOffsetDict[f"mesh{smeshNum}"][0]
            self.smeshPosDict[f"mesh{smeshNum}"][1] += self.smeshOffsetDict[f"mesh{smeshNum}"][1]
            self.smeshPosDict[f"mesh{smeshNum}"][2] += self.smeshOffsetDict[f"mesh{smeshNum}"][2]


    def update(self):

        audio_data = self.stream.read(self.CHUNK, exception_on_overflow = False)

        verts, faces, colors = self.mesh(offset=self.offset, audio_data=audio_data)

        self.mesh1.setMeshData(vertexes=verts, faces=faces, faceColors=colors)
        
        # Not sure how quaternions work but numbers in the arguments change the axis/axes of rotation (scuffed version is 1,2,3,4)
        q = QQuaternion(1, 0, 0, 1).normalized()
        axis, angle = q.getAxisAndAngle()
        self.mesh1.rotate(0.5, axis.x(), axis.y(), axis.z())
        
        #movement of waves in one direction, offset
        self.offset -= 0.05 #Present for 100,50 view

        # Convert into number that Aubio understand.
        volumeData = self.volumeStream.read(self.volumePERIOD_SIZE_IN_FRAME, exception_on_overflow=False)
        #samples = np.fromstring(volumeData, dtype=aubio.float_type) #binary mode of fromstring is deprecated
        samples = np.frombuffer(volumeData, dtype=aubio.float_type)
        # Compute the energy (volume) of the current frame.
        volume = np.sum(samples**2)/len(samples)
        # Format the volume output so it only displays at most six numbers behind 0.
        volume = "{:6f}".format(volume)
        # Finally print the pitch and the volume. print(str(pitch) + " " + str(volume))
        #print(str(int(float(volume)*1000000)))
        self.smesh(volume=volume)

        self.smeshdeletion(removedPos=self.smeshdetection())


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(10)
        self.start()
        self.update()



if __name__ == "__main__":#
    t = Terrain()
    t.animation()