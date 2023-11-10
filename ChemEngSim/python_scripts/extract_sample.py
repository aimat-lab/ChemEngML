import os
import pandas as pd
import datetime as dt
import argparse
import yaml
import shutil


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
    sample_dir = os.path.join(prefix, config["sample_dir"])
    sample_meta = os.path.join(prefix, f"{config['sample_dir']}.h5")
    start_data = os.path.join(prefix, config['package_location'], 'init_sim')
    zfill = config["zfill"]
    
    sample = pd.DataFrame(pd.read_hdf(sample_meta, key='sample'))
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.mkdir(sample_dir)
    
    for idx, row in sample.iterrows():
        dst = os.path.join(sample_dir, row['dir'].split('/')[-1])
        shutil.copytree(row['dir'], dst)
        shutil.copytree(start_data, dst, dirs_exist_ok=True)
