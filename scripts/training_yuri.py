import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from tensorflow.keras import backend as K


from chem_eng_ml.utils.keras import r2
from chem_eng_ml.utils.mlflow import log_array
from chem_eng_ml.data_loader import load_features
from chem_eng_ml.utils.pandas import apply_filter



def standard_cnn(input_shape):
    i = np.random.randint(10000000000)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(7, 7),
                                     activation='relu',
                                     input_shape=input_shape,
                                     name='{}_Conv2D_0'.format(i)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='{}_MaxPooling_0'.format(i)))
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     name='{}_Conv2D_1'.format(i)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='{}_MaxPooling_1'.format(i)))
    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(3, 3),
                                     activation='relu',
                                     name='{}_Conv2D_2'.format(i)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           name='{}_MaxPooling_2'.format(i)))
    model.add(tf.keras.layers.Dropout(0.25, name='{}_Dropout_0'.format(i)))
    model.add(tf.keras.layers.Flatten(name='{}_Flatten_0'.format(i)))
    model.add(tf.keras.layers.Dense(256, activation='relu', name='{}_Dense_0'.format(i)))
    model.add(tf.keras.layers.Dropout(0.5, name='{}_Dropout_1'.format(i)))
    model.add(tf.keras.layers.Dense(2, activation='linear', name='{}_Dense_1'.format(i)))

    return model
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="location of the the ChemEngML directory")
    parser.add_argument("path_to_features", help="path to features")
    parser.add_argument("path_to_labels", help="path to labels")
    args = parser.parse_args()

    #base_dir = args.base_dir
    #conda_env = os.path.join(base_dir, 'environment.yml')
    
    # where to find the data
    feature_path = args.path_to_features
    label_path = args.path_to_labels


    filters = ['temperature_convergence < 0.0000001',
               'pressure_convergence == 0',
               'valid == 1']

    y_trafo = StandardScaler()

    params = dict(model="CNN",
                  y_transform=y_trafo,
                  feature_path=feature_path,
                  label_path=label_path,
                  filters=filters,
                  n_runs=1,
                  loss='mean_squared_error',
                  lr=0.00005,
                  amsgrad=False,
                  n_epochs=40,
                  batch_size=256)
    for key, value in y_trafo.get_params().items():
        params['y_transform_' + key] = value
    params['optimizer'] = tf.keras.optimizers.Adam(lr=params['lr'], amsgrad=params['amsgrad'])
    callbacks = []

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





    train_fraction = 0.02

    n_train = int(train_fraction * y.shape[0])
    metrics = list()

    for n in range(params['n_runs']):
        # splitting
        train_index = np.random.choice(index, n_train, replace=False)
        val_index = np.setdiff1d(index, train_index)

        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        
        # training
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


        # save the model
        # TODO
        
        # analysis
        y_pre = nn.predict(x_val)
        val_r2 = sk_r2(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))
        val_mse = sk_mse(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))
        val_mae = sk_mae(y_trafo.inverse_transform(y_val), y_trafo.inverse_transform(y_pre))

        y_pre = nn.predict(x_test)
        test_r2 = sk_r2(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))
        test_mse = sk_mse(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))
        test_mae = sk_mae(y_trafo.inverse_transform(y_test), y_trafo.inverse_transform(y_pre))
        
        # book keeping
        m = {'true_val_r2': val_r2,
             'true_val_mse': val_mse,
             'true_val_mae': val_mae,
             'true_test_r2': test_r2,
             'true_test_mse': test_mse,
             'true_test_mae': test_mae}
        metrics.append(m)

        K.clear_session()
        

        # save the results
        # TODO
    
    # avarage over the n_runs runs
    for key in metrics[0].keys():
        metric_sum = 0
        for metric in metrics:
            metric_sum += metric[key]
        metric_mean = metric_sum / len(metrics)
        
    # save the avaraged results
    # TODO
        
