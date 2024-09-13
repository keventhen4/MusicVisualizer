import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyqtgraph.opengl import MeshData


def triangle():
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype="uint64")
    return MeshData(vertexes=verts, faces=faces)


def diamant():
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, -1]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype="uint64")
    return MeshData(vertexes=verts, faces=faces)


app = pg.mkQApp()
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example')
w.setCameraPosition(distance=15, azimuth=-90)

g = gl.GLGridItem()
g.scale(2, 2, 1)
w.addItem(g)

triangle_md = triangle()
diamant_md = diamant()
sphere_md = gl.MeshData.sphere(rows=10, cols=20)
cylinder_md = gl.MeshData.cylinder(rows=1, cols=3, radius=[1., 1.], length=1.0, offset=False)

x = np.linspace(-8, 8, 6)

mi1 = gl.GLMeshItem(meshdata=diamant_md, smooth=True, shader='normalColor', glOptions='opaque')
mi1.translate(x[0], 0, 0)
# m2.scale(1, 1, 2)
w.addItem(mi1)

mi2 = gl.GLMeshItem(meshdata=triangle_md, smooth=True, shader='viewNormalColor', glOptions='opaque')
mi2.translate(x[1], 0, 0)
# m3.scale(1, 1, 2)
w.addItem(mi2)

mi3 = gl.GLMeshItem(meshdata=triangle_md, smooth=True, shader='shaded', glOptions='opaque')
mi3.translate(x[2], 0, 0)
# m4.scale(1, 1, 2)
w.addItem(mi3)

mi4 = gl.GLMeshItem(meshdata=sphere_md, smooth=True, color=(1, 0, 0, 0.2), shader='balloon', glOptions='additive')
mi4.translate(x[3], 0, 0)
# m1.scale(1, 1, 2)
w.addItem(mi4)

mi5 = gl.GLMeshItem(meshdata=cylinder_md, smooth=True, shader='heightColor', glOptions='opaque')
#m7.shader()['colorMap'] = np.array([3,2,5])
mi5.translate(x[4], 0, 0)
# m5.scale(1, 1, 2)
w.addItem(mi5)

if __name__ == '__main__':
    pg.exec()
