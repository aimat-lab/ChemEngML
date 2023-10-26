
import numpy as np

xl = 10     # domain length in X-direction
yl = 2      # domain length in Y-direction
nxp = 384   # number of grid points in X-direction
nyp = 129   # number of grid points in Y-direction

xMin = 0
xMax = xl
yMin = 0
yMax = yl

mesh_x = np.linspace(xMin, xMax, nxp, endpoint=True)
#mesh_y = np.linspace(yMin, yMax, nCells_y)
mesh_y = np.array([1-np.cos(np.pi*(i)/(nyp-1)) for i in range(nyp)])

X,Y = np.meshgrid(mesh_x, mesh_y)
# print(gP.X.shape, gP.Y.shape)  # (129, 384) (129, 384)


# Block parameters in terms of number of grid points
plateThickness = 25
blockThickness = [5, 11, 21, 41]
# blockThickness = [21]
blockHeight = [45]
# blockHeight = 60


# flag_arragment = 0 for staggered arrangement 
flag_arragment = 0