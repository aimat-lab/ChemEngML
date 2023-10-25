import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def tf_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


@tf.keras.utils.register_keras_serializable(package='chem_eng_ai', name='LogLearningRateScheduler')
class LogLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """
    Make learning rate schedule function for log reduction.
    Args:
        lr_start (float, optional): Learning rate to start with. The default is 1e-3.
        lr_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epochs (int, optional): Total number of epochs to reduce learning rate towards. The default is 100.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 10.
    Example:
        model.fit(callbacks=[LogLearningRateScheduler()])
    """
    def __init__(self, lr_start=1e-3, lr_stop=1e-5, epochs=100, epomin=10, verbose=0):
        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.epochs = epochs
        self.epomin = epomin
        super(LogLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        if epoch < self.epomin:
            out = self.lr_start
        else:
            out = np.exp(
                float(
                    np.log(self.lr_start) - (np.log(self.lr_start) - np.log(self.lr_stop)) /
                    (self.epochs - self.epomin) * (epoch - self.epomin)
                )
            )
        return float(out)

    def get_config(self):
        config = super(LogLearningRateScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "lr_stop": self.lr_stop, "epochs": self.epochs, "epomin": self.epomin})
        return config


def standard_cnn(input_shape, epochs, **kwargs):
    #i = np.random.randint(10000000000)
    i = "layer"
    model = tf.keras.models.Sequential()
    for n in range(kwargs['n_conv_steps']):
        model.add(
            tf.keras.layers.Conv2D(
                filters=kwargs['n_filters'],
                padding='same',
                kernel_size=(kwargs['kernel_size'], kwargs['kernel_size']),
                activation='relu',
                input_shape=input_shape,
                name=f'conv2d_{n}_{i}'
            )
        )
        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=2,
                strides=2,
                padding='same',
                name=f'max_pooling_{n}_{i}'
            )
        )
    model.add(
        tf.keras.layers.Flatten(
            name=f'flatten_{i}'
        )
    )
    model.add(
        tf.keras.layers.Dense(
            kwargs['dense_size'],
            activation='relu',
            name=f'dense_{i}'
        )
    )
    model.add(
        tf.keras.layers.Dropout(
            kwargs['dropout'],
            name=f'dropout_{i}'
        )
    )
    model.add(
        tf.keras.layers.Dense(
            2,
            activation='linear',
            name=f'output_{i}'
        )
    )

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    lr_scheduler = LogLearningRateScheduler(epochs=epochs, lr_stop=kwargs['final_lr'])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf_r2
        ]
    )
    return model, [lr_scheduler]
