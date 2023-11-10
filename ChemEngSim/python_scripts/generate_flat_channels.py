import os
from matplotlib import pyplot as plt
from grid_generation.gridgen import GridGen
import sys
import numpy as np
import argparse
import yaml


X, Y = np.meshgrid(list(range(384)), [(np.cos(np.pi * i / 129)*-1+1)/2*129 for i in range(129)])


def calc_grid(base_dir, identifier, symmetrical, upper, lower=None):
    path = os.path.join(base_dir, identifier)
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass
    os.mkdir(path)
    gen = GridGen(x_dim=384, y_dim=129, symmetrical=symmetrical)
    grid = gen.flat_channel(upper=upper, lower=lower)
    gen.save(path, identifier)
    gen.to_bin(path)
    plt.contourf(X, Y, grid, 3)
    plt.savefig(os.path.join(path, identifier + '.png'), bbox_inches='tight')


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    base_dir = config["base_dir"]
    zfill = config["zfill"]

    symms = range(40)
    asymms = range(60)
    i = 0
    symmetrical = True
    for upper in symms:
        identifier = str(i).zfill(zfill)
        calc_grid(base_dir, identifier, symmetrical, upper)
        i += 1
    symmetrical = False
    for upper in asymms:
        identifier = str(i).zfill(zfill)
        calc_grid(base_dir, identifier, symmetrical, upper, 0)
        i += 1
    for lower in asymms:
        identifier = str(i).zfill(zfill)
        calc_grid(base_dir, identifier, symmetrical, 0, lower)
        i += 1
    sys.exit(0)
