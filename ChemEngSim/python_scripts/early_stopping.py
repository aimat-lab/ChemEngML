import os
import pandas as pd
import datetime as dt
import argparse
import yaml


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
    zfill = config["zfill"]
    max_time = dt.timedelta(minutes=config["max_time"])
    round_t = config["round_t"]
    round_p = config["round_p"]
    if round_t is None:
        round_t = 9
    if round_p is None:
        round_p = 10
    log_file = os.path.join(prefix, config["log_dir"], f'early_stopping_{config["sample_dir"]}.h5')
    log_file_csv = os.path.join(prefix, config["log_dir"], f'early_stopping_{config["sample_dir"]}.csv')

    print("Storing job statistics to " + log_file)

    calcs = sorted(os.listdir(sample_dir))
    calcs = [os.path.join(sample_dir, calc) for calc in calcs]
    calcs = [calc for calc in calcs if os.path.isdir(calc)]

    if os.path.exists(log_file):
        states = pd.DataFrame(pd.read_hdf(log_file, key='log')).sort_index()
    else:
        states = pd.DataFrame(pd.read_hdf(sample_meta, key='sample')).sort_index()
        states['dir'] = calcs
        states['state'] = "PENDING"
        states['max_time'] = None
        states['start_time'] = None
        states['end_time'] = None
        states['duration'] = None

    n_finished = states[(states['state'] == 'STOPPED') | (states['state'] == 'CONVERGED')].shape[0]
    last = dt.datetime.now()

    while n_finished < len(calcs):
        for i, row in states.iterrows():
            path = states.at[i, 'dir']
            if states.at[i, 'state'] == "PENDING":
                if os.path.isfile(os.path.join(path, 'history.out')):
                    print("Calculation {} has started!".format(str(i).zfill(zfill)))
                    states.at[i, 'state'] = "STARTED"
                    states.at[i, 'max_time'] = dt.datetime.now() + max_time
                    states.at[i, 'start_time'] = dt.datetime.now()

            if states.at[i, 'state'] == "STARTED":
                try:
                    hist = pd.read_fwf(os.path.join(path, 'history.out'), header=None).tail(3).reset_index(drop=True)
                except Exception as e:
                    print(e)
                    continue
                # Wait till history file has three lines
                if hist.shape[0] < 3:
                    continue
                # Check if max time is exceeded
                if states.at[i, 'max_time'] < dt.datetime.now():
                    print("Calculation {} was aborted!".format(str(i).zfill(zfill)))
                    open(os.path.join(path, 'stop.now'), 'a').close()
                    states.at[i, 'state'] = 'STOPPED'
                    states.at[i, 'end_time'] = dt.datetime.now()
                    states.at[i, 'duration'] = states.at[i, 'end_time'] - states.at[i, 'start_time']
                    n_finished += 1
                # Check if simulation has converged
                if hist[1].round(round_p).mean() == hist.at[2, 1].round(round_p) \
                        and hist[3].round(round_t).mean() == hist.at[2, 3].round(round_t):
                    print("Calculation {} converged!".format(str(i).zfill(zfill)))
                    open(os.path.join(path, 'stop.now'), 'a').close()
                    states.at[i, 'state'] = 'CONVERGED'
                    states.at[i, 'end_time'] = dt.datetime.now()
                    states.at[i, 'duration'] = states.at[i, 'end_time'] - states.at[i, 'start_time']
                    n_finished += 1
        # Write the calculation results to file every minute
        if dt.datetime.now() - last > dt.timedelta(minutes=1):
            states.to_hdf(log_file, key='log')
            states.to_csv(log_file_csv)
            last = dt.datetime.now()

    states.to_hdf(log_file, key='log')
    states.to_csv(log_file_csv)
    print('All calculations finished!')
