import tensorflow.keras.backend as K
import mlflow


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


class MlFlowCallback(tf.keras.callbacks.Callback):
    """ This Callback logs train and validation metrics to mlflow on every epoch end.
    """
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(metrics=logs, step=epoch)
