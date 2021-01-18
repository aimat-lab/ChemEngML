import tensorflow as tf
import numpy as np


def standard_cnn(input_shape):
    i = np.random.randint(10000000000)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=(3, 3),
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
