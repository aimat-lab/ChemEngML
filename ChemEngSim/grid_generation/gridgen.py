import numpy as np
from scipy.stats import norm
from scipy.special import binom
from scipy.ndimage.measurements import label
import pandas as pd
import datetime as dt
import os
import h5py


def bernstein(n, k, t):
    return binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    n = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(n):
        curve += np.outer(bernstein(n - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, r=None, numpoints=None):
        self.curve = None
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = numpoints or 10
        r = r or 0.3
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points()

    def calc_intermediate_points(self):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, edgy, r):
    p = np.arctan(edgy) / np.pi + .5
    d = np.diff(points, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    ang = (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    points = np.append(points, np.atleast_2d(ang).T, axis=1)
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], r=r)
        segments.append(seg)
    for i, seg in enumerate(segments):
        if i == 0:
            curve = seg.curve
        else:
            curve = np.concatenate([curve, seg.curve[1:, :]])
    curve = curve.T
    curve[curve < 0] = 0
    return curve


def get_line_segment(x, wall):
    p1 = None
    p2 = None
    for i, loc in enumerate(wall[0]):
        if loc >= x:
            p1 = wall[:, i - 1]
            p2 = wall[:, i]
            break
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - a * p1[0]
    return p1, p2, a, b


class GridGen:
    MAXY = 1024
    AMPL = 75

    def __init__(self, x_dim=384, y_dim=None, spare=0.75, symmetrical=False):
        """Generator for random warped grids for SIMSON

        Args:
            x_dim: x-dimension of the grid
            y_dim: optional, y-dimension of the grid, or a fifth of the channel width
            spare: optional, minimum fraction of the y-dimension to spare for the channel width
            symmetrical: optional, whether to generate a symmetrical channel
        """
        self.x_dim = x_dim
        self.y_dim = y_dim or int(x_dim / 5)
        self.x_borders = np.array(range(self.x_dim+1)).astype(float)
        self.x_centers = np.arange(0.5, self.x_dim, 1.)
        self.y_borders = np.array([(np.cos(np.pi * i / self.y_dim)*-1+1)/2*self.y_dim for i in range(self.y_dim+1)])
        self.y_centers = np.array([(self.y_borders[i] + self.y_borders[i+1])/2 for i in range(self.y_dim)])
        self.spare = spare
        self.grid = None
        self.symmetrical = symmetrical
        self.upper_wall = None
        self.lower_wall = None
        self.generator_parameters = None
        self.fill_fraction = None

    def save(self, prefix, identifier: str):
        assert self.grid is not None, "Grid hasn't been generated yet, nothing to save!"
        time = dt.datetime.now()
        path = os.path.join(prefix, str(identifier)+'.h5')
        general_params = pd.DataFrame(columns=['x_dim', 'y_dim', 'spare', 'symmetrical', 'time'],
                                      index=[str(identifier)],
                                      data=[[self.x_dim, self.y_dim, self.spare, self.symmetrical, time]])
        gen_dict = {key: [val] for key, val in self.generator_parameters.items()}
        generator_params = pd.DataFrame.from_dict(gen_dict)
        generator_params.index = [str(identifier)]
        general_params.to_hdf(path, 'general_parameters')
        generator_params.to_hdf(path, 'generator_parameters')
        f = h5py.File(path, 'a')
        f.create_dataset('grid', data=self.grid)
        if self.generator_parameters['function_name'] == 'random_bezier':
            f.create_dataset('upper_wall', data=self.upper_wall)
            f.create_dataset('lower_wall', data=self.lower_wall)
        f.close()

    def load(self, prefix, identifier):
        path = os.path.join(prefix, str(identifier) + '.h5')
        general_params = pd.DataFrame(pd.read_hdf(path, 'general_parameters'))
        self.x_dim = general_params.at[identifier, 'x_dim']
        self.y_dim = general_params.at[identifier, 'y_dim']
        self.spare = general_params.at[identifier, 'spare']
        self.symmetrical = general_params.at[identifier, 'symmetrical']
        self.generator_parameters = pd.read_hdf(path, 'generator_parameters')
        f = h5py.File(path, 'r')
        self.grid = f['grid'][:]
        self.upper_wall = f['upper_wall'][:]
        self.lower_wall = f['lower_wall'][:]
        f.close()

    def to_bin(self, prefix):
        path = os.path.join(prefix, 'ibm.bin')
        saving_ibm = np.roll(self.grid, int(self.x_dim / 2), axis=1)
        with open(path, 'wb') as f:
            np.array(self.x_dim, dtype=np.uint32).tofile(f)
            np.array(self.y_dim, dtype=np.uint32).tofile(f)
            np.array(self.MAXY, dtype=np.uint32).tofile(f)
            np.array(self.AMPL, dtype=np.float64).tofile(f)
            fortran_data = np.asfortranarray(saving_ibm, 'float64')
            fortran_data.tofile(f)

    def flat_channel(self, upper=0, lower=0):
        self.generator_parameters = dict(function_name='flat_channel',
                                         upper=upper,
                                         lower=lower)
        grid = np.zeros([self.y_dim, self.x_dim], dtype=int)
        if upper > 0:
            grid[:upper] = 1
            if self.symmetrical:
                grid[-upper:] = 1
        if not self.symmetrical:
            if lower > 0:
                grid[-lower:] = 1
        self.grid = grid
        self._calculate_fill_fraction()
        return self.grid

    def random_bezier(self, seed, rad=0.2, edgy=0.3, n=12, x_std=0.05, y_std=0.2):
        """Get a randomized bezier curve as wall structure
        
        Arguments:
            seed(int): Seed for random generator
            rad(float): Radius around the points at which the control point of the bezier curve sit
            edgy(float): The higher, the edgier the curve gets
            n(int): number of points
            x_std(float): Standard deviation for new points in x direction as fraction of the grid width
            y_std(float): Standard deviation for new points in y direction as a fraction of the grid height
        
        Returns:
            grid(np.array): The grid of the channel structure scaled for SIMSON
        """
        
        self.generator_parameters = dict(function_name='random_bezier',
                                         rad=rad,
                                         edgy=edgy,
                                         n=n,
                                         x_std=x_std,
                                         y_std=y_std)
        np.random.seed(seed)
        if self.symmetrical:
            # set build space on every x-coordinate to (1-self.spare)/2
            space = np.ones(self.x_dim)*(self.y_dim*(1-self.spare)/2)
        else:
            # For the first curve set build space on every x-coordinate to 1-self.spare
            space = np.ones(self.x_dim)*np.floor(self.y_dim*(1-self.spare))
        # get n points on the one side of the channel
        points = self._get_random_points(space, n, x_std, y_std)
        # smooth the points to a bezier curve
        self.upper_wall = get_curve(points, edgy, rad)
        # project the curve on the scaled grid
        grid1 = self._curve_2_scaled_grid(self.upper_wall)
        # As the space is only enforced for the points, the bezier interpolation can
        # violate the space to kept free. So apce is enforced again after interpolation
        grid1 = self._enforce_space(space, grid1)
        # for certain cases and after space enforcement, island can occur. Remove them
        grid1 = self._remove_islands(grid1)
        # if the channel should be symmetrica, mirror the the existing wall
        if self.symmetrical:
            self.lower_wall = self.upper_wall.copy()
        # else get the remaining space for the second side by evaluating the grid projection of the first wall
        else:
            space = self._get_space(grid1)
            points = self._get_random_points(space, n, x_std, y_std)
            self.lower_wall = get_curve(points, edgy, rad)
        # wirte the second wall into a new grid and "clean" it
        grid2 = self._curve_2_scaled_grid(self.lower_wall)
        grid2 = self._enforce_space(space, grid2)
        grid2 = self._remove_islands(grid2)
        # merge the grids into one
        self.grid = grid1 + np.flip(grid2, axis=0)
        # calculate how much of the available space is filled
        self._calculate_fill_fraction()
        # calculate the surface of the channel
        self._calculate_surfaces()
        return self.grid

    def _get_random_points(self, space, n=12, x_std=0.05, y_std=0.2):
        # xpoints contrains the x and y coordinates of tghe supporting
        # points for the bezier curve
        points = np.zeros([n + 1, 2])
        _x_dim = self.x_dim/n
        x_dist = dict(loc=0, scale=x_std * _x_dim, size=1)
        y_dist = dict(loc=space[0] / 2, scale=y_std * space[0], size=1)
        # generate a first y coordinate
        y = norm.rvs(**y_dist) # TODO this should be sanitized to be between 0 and space
        # set first and las point to the same y coordinate
        points[0, :] = [0, y]
        points[-1, :] = [self.x_dim, y]

        for i in range(1, n):
            _x = int(np.round(i * self.x_dim / n))
            x_dist['loc'] = i * self.x_dim / n
            y_dist['loc'] = y
            y_dist['scale'] = y_std * space[_x]
            if i == n:
                x = self.x_dim - 1
            else:
                x = max(0, min(self.x_dim - 1, norm.rvs(**x_dist)))
            y = max(0, min(space[int(np.round(x))], norm.rvs(**y_dist)))
            points[i, :] = [x, y]
        return points

    def _get_space(self, grid):
        space = np.zeros(self.x_dim)
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if grid[y, x] == 1:
                    space[x] = max(0, self.y_dim*(1-self.spare) - self.y_centers[y])
        return space

    def _find_y_loc(self, y):
        return np.searchsorted(self.y_borders, y, side="left")

    def _enforce_space(self, space, grid):
        for x in range(self.x_dim):
            y = self._find_y_loc(space[x])
            grid[y:, x] = 0
        return grid

    def _curve_2_scaled_grid(self, wall):
        grid = np.zeros([self.y_dim, self.x_dim], dtype=int)

        # Get pixel trace
        for n in range(len(wall[0])-1):
            # print("n: ", n)
            p0 = wall[:, n]
            p1 = wall[:, n+1]
            # swap if we go backwards
            if p0[0] > p1[0]:
                _p0 = p0.copy()
                p0 = p1.copy()
                p1 = _p0

            # Get start and end x coords for segment
            start_x = int(p0[0])
            end_x = int(p1[0])+1

            # Get line equation for segment
            a = (p1[1]-p0[1])/(p1[0]-p0[0])
            b = p0[1]-a*p0[0]

            for loc, x in enumerate(self.x_borders[start_x:end_x]):
                # For the first y get the y coordinate from points
                if loc == 0:
                    y0 = self._find_y_loc(p0[1])
                # Else interpolate
                else:
                    y0 = self._find_y_loc(max(0, a*x+b))
                # For the last y get the y coordinate from points
                if loc == len(self.x_borders[start_x:end_x])-1:
                    y1 = self._find_y_loc(p1[1])
                # Else interpolate
                else:
                    y1 = self._find_y_loc(max(0, a*(x+1)+b))
                # y0 = min(self.y_dim-1, max(0, y0))
                # y1 = min(self.y_dim-1, max(0, y1))
                x = int(x)
                if y0 == y1:
                    grid[y0, x:x+1] = 1
                else:
                    if a >= 0:
                        grid[y0:y1, x:x+1] = 1
                    else:
                        grid[y1:y0, x:x+1] = 1

        padded_grid = np.zeros([self.y_dim+1, self.x_dim+2], dtype=int)
        padded_grid[1:, 1:-1] = grid
        padded_grid[0, :] = 1
        padded_grid[:, 0] = 1
        padded_grid[:, -1] = 1

        # Fill grid
        is_void = padded_grid == 0
        labels, n = label(is_void)
        on_border = set(labels[:, 0]) | set(labels[:, -1]) | set(labels[0, :]) | set(labels[-1, :])
        for lab in range(1, n + 1):
            if lab not in on_border:
                padded_grid[labels == lab] = 1

        return padded_grid[1:, 1:-1]

    def _remove_islands(self, grid):
        # Use scipy.ndimage.measurements.label to label all connected areas in the grid:
        labels, n = label(grid)
        # if there are more than two areas:
        if n > 1:
            # get the labels of all areas that are connected to an outer border of the grid
            on_border = set(labels[:, 0]) | set(labels[:, -1]) | set(labels[0, :]) | set(labels[-1, :])
            for l in range(1, n + 1):
                # set the areas that are not connected to the borders of the grid to zero (empty channel)
                if l not in on_border:
                    grid[labels == l] = 0
        return grid

    def _calculate_fill_fraction(self):
        assert self.grid is not None, "You first need to generate a grid."
        ys = [(np.cos(np.pi * i / self.y_dim) * -1 + 1) / 2 * self.y_dim for i in range(self.y_dim + 1)]
        y_dists = np.array([ys[i + 1] - ys[i] for i in range(self.y_dim)])
        scales = np.zeros([self.y_dim, self.x_dim])
        for i in range(self.x_dim):
            scales[:, i] = y_dists
        max_scale = scales.sum()
        self.fill_fraction = np.multiply(self.grid, scales).sum() / max_scale
        self.generator_parameters['area'] = self.fill_fraction

    def _calculate_surfaces(self):
        assert self.upper_wall is not None and self.lower_wall is not None, "You first need to generate the walls."
        self.upper_surf = self._get_wall_length(self.upper_wall)/self.x_dim
        self.lower_surf = self._get_wall_length(self.lower_wall)/self.x_dim
        self.generator_parameters['upper_surface'] = self.upper_surf
        self.generator_parameters['lower_surface'] = self.lower_surf
        self.generator_parameters['total_surface'] = self.upper_surf + self.lower_surf
        self.generator_parameters['surface_ratio'] = (max(self.lower_surf, self.upper_surf) /
                                                      min(self.lower_surf, self.upper_surf))

    def _get_wall_length(self, wall):
        length = 0
        for i in range(wall.shape[1]-1):
            p0 = wall[:, i]
            p1 = wall[:, i+1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            length += np.sqrt(dx**2+dy**2)
        return length
