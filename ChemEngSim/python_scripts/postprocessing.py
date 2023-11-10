import os
import sys
import datetime as dt
import argparse

import yaml
import pandas as pd
from tqdm import tqdm
import numpy as np

from postprocess.postprocessing import postprocess


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()
    prefix = os.environ['SLURM_SUBMIT_DIR']
    config = load_config(os.path.join(prefix, args.config))

    base_dir = os.path.join(prefix, config["sample_dir"])
    round_t = config["round_t"]
    round_p = config["round_p"]
    if round_t is None:
        round_t = 9
    if round_p is None:
        round_p = 10

    calcs = sorted(os.listdir(base_dir))
    calcs = [calc for calc in calcs if os.path.isdir(os.path.join(base_dir, calc))]

    label_cols = {"Cf": float, "St": float}
    conv_cols = {'p_conv': float, 't_conv': float}
    early_stopping_cols = {'state': str,
                           'max_time': '<M8[ns]',
                           'start_time': '<M8[ns]',
                           'end_time': '<M8[ns]',
                           'duration': '<m8[ns]'}
    meta_cols = {'dir': str}
    dtypes = {**label_cols, **conv_cols, **early_stopping_cols, **meta_cols}
    columns = (list(label_cols.keys()) +
               list(conv_cols.keys()) +
               list(early_stopping_cols.keys()) +
               list(meta_cols.keys()))
    results = pd.DataFrame(columns=columns).astype(dtypes)
    results.index.rename('id', inplace=True)

    log_file = pd.DataFrame(
        pd.read_hdf(
            os.path.join(
                prefix,
                config["log_dir"],
                f'early_stopping_{config["sample_dir"]}.h5'
            ),
            key='log'
        )
    )

    for calc in tqdm(calcs):
        idx = int(calc)
        path = os.path.join(base_dir, calc)

        if "stop.now" in os.listdir(path):
            # Get labels
            try:
                Cf, St = postprocess(path)
            except ValueError as e:
                print(f'Postprocessing for {idx} failed. Strange structure?\n{e}')
                Cf = np.nan
                St = np.nan
            # Get convergence
            hist = pd.read_fwf(os.path.join(path, 'history.out'), header=None).tail(3).reset_index(drop=True)
            p_conv = np.abs(hist[1].round(round_p).mean() - hist.at[2, 1].round(round_p))
            t_conv = np.abs(hist[3].round(round_t).mean() - hist.at[2, 3].round(round_t))
            # Compose row
            row = (pd.Series({'Cf': Cf, 'St': St})
                   .append(pd.Series({'p_conv': p_conv, 't_conv': t_conv}))
                   .append(log_file.loc[idx, 'state':'duration'])
                   .append(pd.Series({'dir': path}))
                   )
            results.loc[idx, :] = row

    results.to_hdf(base_dir+'_'+'results.h5', 'results')
    results.to_csv(base_dir+'_'+'results.csv')

    sys.exit(0)
