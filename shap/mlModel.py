import os 
import tensorflow as tf
from pickle import load

def tf_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true-K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def ML_prediction(x1_copy):
    model, y_scaler = loadMLmodel()
    # Prediction of the Cf and St values using the trained ML model
    y_untrained_pred = model.predict(x1_copy)   
    # print(y_untrained_pred.shape)
    Cf=y_scaler.inverse_transform(y_untrained_pred)[:,0]
    St=y_scaler.inverse_transform(y_untrained_pred)[:,1]
    
    return Cf, St


def loadMLmodel():    
    # model_base_path = '/net/istmmerkur/localhome/hi222/uploads/ML/44986783'
    model_base_path = '/net/istmmerkur/localhome/hi222/uploads/retrain_cnn_Yuri_Final'
    model_path = os.path.join(model_base_path, 'full_model_model')
    scaler_path = os.path.join(model_base_path, 'y_scaler.npy')

    model = tf.keras.models.load_model(model_path, custom_objects={'tf_r2': tf_r2})

    with open(scaler_path, 'br') as f:
        y_scaler = load(f)
        
    return model, y_scaler
