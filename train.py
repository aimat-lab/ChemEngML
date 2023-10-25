import argparse
import gc
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from pickle import dump
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
import tensorflow.keras.backend as K
import yaml
import tensorflow as tf

from chem_eng_ai.architectures import standard_cnn


METRIC_FUNCTIONS = dict(r2=sk_r2, mae=sk_mae, mse=sk_mse)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument("--testing", type=str2bool, nargs='?', const=True, default=False, help="Activate testing mode.")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config, args.testing


def _get_trainable_parameters(model):
    return np.sum([K.count_params(w) for w in model.trainable_weights])


def _get_columns():
    metric_columns = [f'{key}_Cf' for key in METRIC_FUNCTIONS.keys()]
    metric_columns.extend([f'{key}_St' for key in METRIC_FUNCTIONS.keys()])
    true_columns = [f'{key}_Cf_true' for key in METRIC_FUNCTIONS.keys()]
    true_columns.extend([f'{key}_St_true' for key in METRIC_FUNCTIONS.keys()])
    columns = metric_columns + true_columns
    return columns, metric_columns, true_columns


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_x, label_y, batch_size=2, shuffle=True):
        # samples and labels should be numpy arrays (or tensors)

        self.image_x = image_x
        self.label_y = label_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_x))
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        num_images = len(self.image_x)
        num_batches = int(np.floor(num_images / self.batch_size))
        if num_batches<1:
            exit("ERROR: the batch size of %i is too large for a dataset with %i samples"%(self.batch_size, num_images))
        return num_batches

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        #print("Batch %i: %i samples"%(index+1, len(indices)))
        if len(indices)<=2:
            exit("ERROR: batch size (%i) too small to split in two subsets"%(len(indices)))

        # split the indices randomly
        np.random.shuffle(indices)
        split = np.split(indices, 2, axis=0)
        batch_inputs_to_flip = self.image_x[split[0]]
        batch_labels_flipped = self.label_y[split[0]]
        batch_inputs_to_shift = self.image_x[split[1]]
        batch_labels_shifted = self.label_y[split[1]]

        # flipping batch_inputs_to_flip:
        batch_flipped = np.flip(batch_inputs_to_flip, axis=1)

        # shifting batch_inputs_to_shift
        batch_shifted = []
        for counter, i in enumerate(np.random.randint(1, batch_inputs_to_shift.shape[2], batch_inputs_to_shift.shape[0])):# max_shift=20...(50/384)
            batch_shifted.append(np.roll(batch_inputs_to_shift[counter], i, axis=1))
        batch_shifted = np.array(batch_shifted)

        inputs_combined = np.concatenate([batch_flipped, batch_shifted])
        labels_combined = np.concatenate([batch_labels_flipped, batch_labels_shifted])

        return inputs_combined, labels_combined

    def on_epoch_end(self):
        "Updates indexes after each epoch"

        if self.shuffle == True:
            np.random.shuffle(self.indices)


def evaluate(name, output_path, X_train, y_train, X_val, y_val, model_kwargs, epochs, y_scaler, results, batch_size=256):
    model, callbacks = standard_cnn(X_train[0].shape, epochs=epochs, **model_kwargs)
    history = model.fit(DataGenerator(X_train, y_train, batch_size=batch_size),
        validation_data = DataGenerator(X_val, y_val, batch_size=batch_size),
        epochs=epochs,
        callbacks=callbacks
    )

    y_pred = model.predict(X_val)
    y_pred_true = y_scaler.inverse_transform(y_pred)
    y_test_true = y_scaler.inverse_transform(y_val)

    for key, func in METRIC_FUNCTIONS.items():
        results.loc[name, f'{key}_Cf'] = func(y_val[:, 0], y_pred[:, 0])
        results.loc[name, f'{key}_St'] = func(y_val[:, 1], y_pred[:, 1])
        results.loc[name, f'{key}_Cf_true'] = func(y_test_true[:, 0], y_pred_true[:, 0])
        results.loc[name, f'{key}_St_true'] = func(y_test_true[:, 1], y_pred_true[:, 1])
    if 'trainable_parameters' in results.columns:
        results.loc[name, 'trainable_parameters'] = np.sum([K.count_params(w) for w in model.trainable_weights])
    results.to_csv(os.path.join(output_path, 'results.csv'))

    # Saving stuff
    with open(os.path.join(output_path, f'{name}_meta_data.npy'), 'bw') as f:
        dump(history.history, f)
    model.save(os.path.join(output_path, f'{name}_model'))

    hist = history.history
    df = pd.DataFrame(index=list(range(epochs)), columns=hist.keys())
    for key in hist.keys():
        df.loc[:, key] = hist[key]
    df.to_csv(os.path.join(output_path, f'{name}_history.csv'))

    valid_keys = [key for key in hist.keys() if 'val_' not in key]
    for key in valid_keys:
        fig, ax = plt.subplots()
        ax.plot(range(epochs), hist[key])
        try:
            ax.plot(range(epochs), hist[f"val_{key}"])
        except KeyError:
            pass
        plt.savefig(os.path.join(output_path, f"{name}_{key}.png"), bbox_inches='tight')
        plt.close(fig)
    return results


def load_data(dataset_path):
    with open(os.path.join(dataset_path, 'X_train.npy'), 'br') as f:
        X = np.load(f)
    with open(os.path.join(dataset_path, 'X_test.npy'), 'br') as f:
        X_test = np.load(f)
    with open(os.path.join(dataset_path, 'y_train.npy'), 'br') as f:
        y = np.load(f)
    with open(os.path.join(dataset_path, 'y_test.npy'), 'br') as f:
        y_test = np.load(f)
    with open(os.path.join(dataset_path, 'x_flat.npy'), 'br') as f:
        x_flat = np.load(f)
    with open(os.path.join(dataset_path, 'y_flat.npy'), 'br') as f:
        y_flat = np.load(f)

    X = np.concatenate((X, x_flat[5:]))
    X_test = np.concatenate((X_test, x_flat[:5]))
    y = np.concatenate((y, y_flat[5:]))
    y_test = np.concatenate((y_test, y_flat[:5]))

    return X, X_test, y, y_test


def test(config):
    return train(config, testing=True)


def train(config, testing=False):
    dataset_path = config['dataset_path']
    output_path = config['output_path']
    model_kwargs = config['suggestion']
    epochs = 100
    batch_size = 32
    n_folds = 5
    X, X_test, y, y_test = load_data(dataset_path)

    # adjust dataset size
    train_indexes = np.arange(len(X))
    np.random.shuffle(train_indexes)
    test_indexes = np.arange(len(X_test))
    np.random.shuffle(test_indexes)
    X = X[train_indexes]
    y = y[train_indexes]
    X_test = X_test[test_indexes]
    y_test = y_test[test_indexes]


    # overwrite some settings in test mode
    if testing:
        print("Running in test mode")
        X = X[:500]
        X_test = X_test[:500]
        y = y[:500]
        y_test = y_test[:500]
        batch_size = 6
        n_folds = 2
        epochs = 5
        print("Test mode: Use only %i samples for training and %i samples for testing"%(len(X), len(y)))
        print("Test mode: Use a batch size of %i, %i fold cross validation and %i epochs"%(batch_size, n_folds, epochs))

    print("Dataset size:")
    print("Training data:")
    print(X.shape, y.shape)
    
    print("Testing data:")
    print(X_test.shape, y_test.shape)
    

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    y_test_scaled = y_scaler.transform(y_test)

    # Save the scaler
    with open(os.path.join(output_path, 'y_scaler.npy'), 'bw') as f:
        dump(y_scaler, f)

    cv = KFold(n_splits = n_folds, shuffle=True)

    columns, metric_columns, true_columns = _get_columns()
    columns.append('trainable_parameters')
    metric_columns.append('trainable_parameters')

    cv_index = [f'cv_{i}' for i in range(n_folds)]
    cv_index.extend(['mean', 'median', 'full_model'])
    results = pd.DataFrame(columns=columns, index=cv_index)


    evaluate('full_model', output_path, X, y_scaled, X_test, y_test_scaled, model_kwargs, epochs, y_scaler,
             results, batch_size=batch_size)
    results.loc['mean', :] = results.loc[cv_index, :].mean(axis=0)
    results.loc['median', :] = results.loc[cv_index, :].median(axis=0)
    results.to_csv(os.path.join(output_path, 'results.csv'))

    sigopt_returns = list()
    for key in metric_columns:
        sigopt_returns.append({'name': key, 'value': results.at['mean', key]})

    return sigopt_returns, None


if __name__ == "__main__":
    config, testing = parse_config()
    results, meta_data = train(config, testing=testing)
    print(results)
    print('\n')
    print(meta_data)
    sys.exit(0)
