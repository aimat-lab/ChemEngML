import argparse
import os
import yaml
from tqdm import tqdm
import pandas as pd
import h5py
import sys
import shutil
import numpy as np
from scipy.ndimage import correlate


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()
    prefix = os.environ['SLURM_SUBMIT_DIR']
    with open(os.path.join(prefix, args.config)) as f:
        return prefix, yaml.safe_load(f)


if __name__ == "__main__":
    prefix, config = load_config()
    base_dir = os.path.join(prefix, config["base_dir"])

    calcs = sorted(os.listdir(base_dir))
    calcs = [calc for calc in calcs if os.path.isdir(os.path.join(base_dir, calc))]
    meta_cols = {'dir': str, 'id': str}

    h5_path = os.path.join(os.path.join(base_dir, calcs[0]), calcs[0] + '.h5')
    general = pd.DataFrame(pd.read_hdf(h5_path, 'general_parameters'))
    generator = pd.DataFrame(pd.read_hdf(h5_path, 'generator_parameters'))
    general_cols = dict(general.dtypes)
    generator_cols = dict(generator.dtypes)
    dtypes = {**general_cols, **generator_cols, **meta_cols}
    columns = list(general_cols.keys()) + list(generator_cols.keys()) + list(meta_cols.keys())

    out_path = os.path.join(prefix, config["base_dir"] + "_post_generation")
    os.mkdir(out_path)
    png_path = os.path.join(out_path, 'pngs')
    os.mkdir(png_path)
    results = pd.DataFrame(columns=columns).astype(dtypes)
    feature_file = h5py.File(os.path.join(out_path, 'features.h5'), 'w')

    y_dim = 129
    y_borders = np.array([(np.cos(np.pi * i / y_dim) * -1 + 1) / 2 * y_dim for i in range(y_dim + 1)])
    y_diffs = [y_borders[i + 1] - y_borders[i] for i in range(y_dim)]

    for calc in tqdm(calcs):
        idx = int(calc)
        path = os.path.join(base_dir, calc)

        # Get labels and metadata:
        h5_path = os.path.join(path, calc + '.h5')
        general = pd.DataFrame(pd.read_hdf(h5_path, 'general_parameters'))[general_cols]
        generator = pd.DataFrame(pd.read_hdf(h5_path, 'generator_parameters'))[generator_cols]
        row = (general.iloc[0, :]
               .append(generator.iloc[0, :])
               .append(pd.Series({'dir': path, 'id': general.index.values[0]}))
               )
        results.loc[idx, :] = row

        # Get grids
        f = h5py.File(h5_path, 'r')
        grid = f['grid'][:]
        f.close()
        feature_file.create_dataset(str(idx), data=grid)

        # Calculate correct surface increase
        inv_grid = np.zeros((grid.shape[0] + 4, grid.shape[1] + 2))
        inv_grid[2:-2, 1:-1] = ((grid - 1) * -1).astype(float)
        inv_grid[:, 0] = inv_grid[:, 1]
        inv_grid[:, -1] = inv_grid[:, -2]
        lower = inv_grid[0:68, :]
        upper = np.flipud(inv_grid)[0:68, :]

        u_surf = 0.
        l_surf = 0.
        for y in range(0, 68 - 2):
            # Material is 0, void is 1:
            # If the kernel is on void, the center produces a -9 pulling everything below 0
            # If the kernel is on material, bordering void is multiplied by the current y distance for vertical borders
            # and multiplied by 1 for horizontal borders...
            # This overestimates the surface.
            kernel = np.array(
                [
                    [0, 1, 0],
                    [y_diffs[y], -9, y_diffs[y]],
                    [0, 1, 0]
                ]
            )
            u_surf += np.maximum(0., correlate(upper[y:y + 3, :], kernel)).sum()
            l_surf += np.maximum(0., correlate(lower[y:y + 3, :], kernel)).sum()

        results.loc[idx, 'lower_surface'] = l_surf
        results.loc[idx, 'upper_surface'] = u_surf
        results.loc[idx, 'total_surface'] = l_surf + u_surf
        results.loc[idx, 'surface_ratio'] = np.min([l_surf, u_surf]) / np.max([l_surf, u_surf])

        # Copy images
        src = os.path.join(path, calc + '.png')
        dst = os.path.join(png_path, calc + '.png')
        shutil.copyfile(src, dst)

    feature_file.close()
    results.to_hdf(os.path.join(out_path, 'statistics.h5'), 'statistics')
    results.to_csv(os.path.join(out_path, 'statistics.csv'))

    sys.exit(0)
