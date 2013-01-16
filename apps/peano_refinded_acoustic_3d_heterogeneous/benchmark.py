import FractalTerrain as FT
from pylab import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def generate_landscape_layers(num_layers, XI, XF, size):
    iters = int(np.log2(size - 1))
    width = np.fabs(XF - XI)
    layer_width = width / num_layers
    layers_data = []
    for layer_counter in range(num_layers):
        # generate the landscape
        terrain = FT.FractalTerrain(iterations=iters, seed=10.0, deviation=5.0, roughness=3)
        height = np.array(terrain.Vertices)

        # normalaize the landscape
        # this part needs improvement
        height =  np.fabs(height)
        height /= amax(np.fabs(height))
        li = XI + layer_counter * layer_width
        height += li
        height *= layer_width 
        # assert amax(height) <= li + layer_width

        layers_data.append(height)
    return layers_data


def vis_landscape(landscape,XI, XF, size, num_layers):
    fig = figure()
    ax = Axes3D(fig)
    X = np.linspace( XI, XF, num=size)
    Y = np.linspace( XI, XF, num=size)
    X, Y = np.meshgrid(X, Y)
    hold(True)
    for ls in landscape:
        ax.plot_surface(X, Y, ls, rstride=1, cstride=1, cmap=cm.jet)
    show()

