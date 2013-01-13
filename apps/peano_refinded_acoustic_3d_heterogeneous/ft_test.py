import FractalTerrain as FT
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__':
    terrain = FT.FractalTerrain(iterations=6, seed=10.0, deviation=5.0, roughness=3)
    height = np.array(terrain.Vertices)

    fig = figure()
    ax = Axes3D(fig)
    X = np.linspace(2.0, 3.0, num=terrain.Size)
    Y = np.linspace(2.0, 3.0, num=terrain.Size)
    X, Y = np.meshgrid(X, Y)
    Z = height
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
    show()

    
