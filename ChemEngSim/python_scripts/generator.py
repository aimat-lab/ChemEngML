from grid_generation import GridGen
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
from matplotlib import pyplot as plt
import yaml
import argparse


X, Y = np.meshgrid(list(range(384)), [(np.cos(np.pi * i / 129)*-1+1)/2*129 for i in range(129)])


def calc_grid(base_dir, identifier, seed, spare, symmetrical, n, rad, edgy, x_std, y_std):
    path = os.path.join(base_dir, identifier)
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass
    os.mkdir(path)
    gen = GridGen(x_dim=384, y_dim=129, spare=spare, symmetrical=symmetrical)
    np.random.seed(seed)
    success = False
    while not success:
        try:
            seed = np.random.randint(987654321),
            grid = gen.random_bezier(seed=seed, n=n, rad=rad, edgy=edgy, x_std=x_std, y_std=y_std)
            success = True
        except:
            print(f"Structure {identifier} was strange, doing it again.")
            print(f"Parameters: seed={seed}, n={n}, rad={rad}, edgy={edgy}, x_std={x_std}, y_std={y_std}")
            pass

    gen.save(path, identifier)
    gen.to_bin(path)
    plt.contourf(X, Y, grid, 3)
    plt.savefig(os.path.join(path, identifier+'.png'), bbox_inches='tight')


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()
    prefix = os.environ['SLURM_SUBMIT_DIR']
    config = load_config(os.path.join(prefix, args.config))

    base_dir = os.path.join(prefix, config["base_dir"])
    zfill = config["zfill"]
    n_processes = config["n_processes"]
        
    ns = config["ns"]
    symmetricals = config["symmetricals"]
    rads = config["rads"]
    edgys = config["edgys"]
    x_stds = config["x_stds"]
    y_stds = config["y_stds"]
    spare = config["spare"]
    n_per_var = config["n_per_var"]

    n_var = len(ns)*len(symmetricals)*len(rads)*len(edgys)*len(x_stds)*len(y_stds)
    n_total = n_var*n_per_var
    print("Calculating {} random structures in {} parallel processes.".format(n_total, n_processes))

    with mp.Pool(n_processes) as pool:
        count = config["start_count"]
        for n in ns:
            for symmetrical in symmetricals:
                for rad in rads:
                    for edgy in edgys:
                        for x_std in x_stds:
                            x_std = 1/(x_std*n)
                            for y_std in y_stds:
                                args_list = []
                                for i in range(n_per_var):
                                    identifier = str(count).zfill(zfill)
                                    # This seed is always different as it is calculated always
                                    # with a different system time
                                    args = (base_dir,
                                            identifier,
                                            np.random.randint(987654321),
                                            spare,
                                            symmetrical,
                                            n,
                                            rad,
                                            edgy,
                                            x_std,
                                            y_std)
                                    args_list.append(args)
                                    count += 1
                                pool.starmap(calc_grid, args_list)

    print(count)
