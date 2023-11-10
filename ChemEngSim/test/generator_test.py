from grid_generation.gridgen import GridGen
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime as dt
import pandas as pd
import os
import shutil


if __name__ == '__main__':
    times = list()
    testdir = './test_exports/'
    if os.path.exists(testdir):
        shutil.rmtree(testdir)
    os.mkdir(testdir)
    for i in range(5):
        seed = np.random.randint(99999999)
        # print(seed)
        start = dt.now()
        gen = GridGen(x_dim=384, y_dim=129, spare=0.5, symmetrical=False)
        grid = gen.random_bezier(seed=seed, n=2, rad=.2, edgy=.2, x_std=.3, y_std=.1)
        gen.save(testdir, f'test_{i}')
        end = dt.now()
        gen_params = pd.read_hdf(os.path.join(testdir, f'test_{i}.h5'), 'generator_parameters')
        print(gen_params['area'].values[0])
        times.append(end-start)
    print(f"Time: {np.sum(times)/50}")
    X, Y = np.meshgrid(list(range(384)), [(np.cos(np.pi * i / 129) * -1 + 1) / 2 * 129 for i in range(129)])
    start = dt.now()
    plt.contourf(X, Y, np.flip(grid, axis=0), 3)
    end = dt.now()
    print(f"Plotting time: {end-start}")
    plt.show()

