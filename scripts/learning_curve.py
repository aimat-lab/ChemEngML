import argparse
import os
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from tensorflow.keras import backend as K

from chem_eng_ml.utils.keras import MlFlowCallback
from chem_eng_ml.utils.keras import r2
from chem_eng_ml.utils.mlflow import log_array
from chem_eng_ml.data_loader import load_features
from chem_eng_ml.architectures import standard_cnn
from chem_eng_ml.utils.pandas import apply_filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="location of the the ChemEngML directory")
    parser.add_argument("path_to_features", help="path to features")
    parser.add_argument("path_to_labels", help="path to labels")
    args = parser.parse_args()

    base_dir = args.base_dir
    conda_env = os.path.join(base_dir, 'environment.yml')
    feature_path = args.path_to_features
    label_path = args.path_to_labels

    experiment_name = "ChemEngML-LC"
    mlflow.set_experiment(experiment_name)

    filters = ['temperature_convergence < 0.0000001',
               'pressure_convergence == 0',
               'valid == 1']

    y_trafo = StandardScaler()

    params = dict(model="CNN",
                  y_transform=y_trafo,
                  feature_path=feature_path,
                  label_path=label_path,
                  filters=filters,
                  n_runs=5,
                  loss='mean_squared_error',
                  lr=0.00005,
                  amsgrad=False,
                  n_epochs=40,
                  batch_size=256)
    for key, value in y_trafo.get_params().items():
        params['y_transform_' + key] = value
    params['optimizer'] = tf.keras.optimizers.Adam(lr=params['lr'], amsgrad=params['amsgrad'])
    callbacks = [MlFlowCallback()]

    # Load and filter dataset
    df = pd.DataFrame(pd.read_hdf(params['label_path'], 'results'))
    for f in filters:
        df = apply_filter(df, f)

    filtered_index = df.index
    y = df.loc[:, ['Cf', 'St']].values
    y = y_trafo.fit_transform(y)
    x = load_features(params['feature_path'], filtered_index)

    index = np.array(list(range(y.shape[0])))
    test_index = np.random.choice(index, 1000, replace=False)
    index = np.setdiff1d(index, test_index)

    x_test = x[test_index]
    y_test = y[test_index]

    run_id = str(np.random.randint(99999)).zfill(5)

    with mlflow.start_run(nested=True, run_name=run_id):
        log_array(test_index, 'test_index')
        mlflow.sklearn.log_model(sk_model=y_trafo,
                                 artifact_path='y_transform',
                                 conda_env=conda_env)
        mlflow.log_params(params)

        for train_fraction in np.arange(0.05, 0.95, 0.05):
            train_fraction = np.round(train_fraction, 2)
            with mlflow.start_run(nested=True,
                                  run_name="{}_{}".format(run_id, train_fraction)):
                n_train = int(train_fraction * y.shape[0])
                mlflow.log_param('train_fraction', train_fraction)
                mlflow.log_param('n_train_samples', n_train)
                metrics = list()

                for n in range(params['n_runs']):
                    with mlflow.start_run(nested=True,
                                          run_name="{}_{}_{}".format(run_id, train_fraction, n)):
                        mlflow.log_param('train_fraction', train_fraction)
                        mlflow.log_param('n_train_samples', n_train)

                        train_index = np.random.choice(index, n_train, replace=False)
                        val_index = np.setdiff1d(index, train_index)
                        mlflow.log_param('n', n)

                        log_array(train_index, 'train_index')

                        x_train, x_val = x[train_index], x[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        nn = standard_cnn(x[0].shape)
                        nn.compile(optimizer=params['optimizer'],
                                   loss=params['loss'],
                                   metrics=[tf.keras.metrics.MeanAbsoluteError(),
                                            r2])

                        nn.fit(x_train, y_train,
                               epochs=params['n_epochs'],
                               batch_size=params['batch_size'],
                               validation_data=(x_val, y_val),
                               callbacks=callbacks)

                        y_pre = nn.predict(x_val)
                        val_r2 = sk_r2(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))
                        val_mse = sk_mse(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))
                        val_mae = sk_mae(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))

                        y_pre = nn.predict(x_test)
                        test_r2 = sk_r2(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))
                        test_mse = sk_mse(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))
                        test_mae = sk_mae(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))

                        m = {'true_val_r2': val_r2,
                             'true_val_mse': val_mse,
                             'true_val_mae': val_mae,
                             'true_test_r2': test_r2,
                             'true_test_mse': test_mse,
                             'true_test_mae': test_mae}
                        metrics.append(m)

                        mlflow.log_metrics(m)

                        mlflow.keras.log_model(keras_model=nn,
                                               artifact_path="model",
                                               conda_env=conda_env,
                                               keras_module=tf.keras,
                                               save_format="tf")
                        K.clear_session()

                    for key in metrics[0].keys():
                        metric_sum = 0
                        for metric in metrics:
                            metric_sum += metric[key]
                        mlflow.log_metric(key='mean_' + key, value=metric_sum / len(metrics))
