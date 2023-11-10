import os
import sys
import pandas as pd
from tqdm import tqdm
import h5py
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="base directory")
    args = parser.parse_args()
    base_dir = args.dir

    calcs = sorted(os.listdir(base_dir))
    calcs = [calc for calc in calcs if os.path.isdir(os.path.join(base_dir, calc))]

    meta_cols = {'dir': str, 'id': str}

    h5_path = os.path.join(os.path.join(base_dir, calcs[0]), calcs[0] + '.h5')
    general = pd.DataFrame(pd.read_hdf(h5_path, 'general_parameters'))
    generator = pd.DataFrame(pd.read_hdf(h5_path, 'generator_parameters'))
    general_cols = dict(general.dtypes)
    generator_cols = dict(generator.dtypes)
    dtypes = {**general_cols, **generator_cols, **meta_cols}
    columns = list(list(general_cols.keys()) + list(generator_cols.keys()) + list(meta_cols.keys()))

    # general_cols = {'x_dim': int, 'y_dim': int, 'spare': float, 'symmetrical': bool}
    # generator_cols = {'function_name': str, 'rad': float, 'edgy': float, 'n': int, 'x_std': float, 'y_std': float}

    results = pd.DataFrame(columns=columns).astype(dtypes)

    feature_file = h5py.File(base_dir+'_'+'features.h5', 'w')

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

    feature_file.close()
    results.to_hdf(base_dir+'_'+'results.h5', 'results')
    results.to_csv(base_dir+'_'+'results.csv')

    sys.exit(0)
