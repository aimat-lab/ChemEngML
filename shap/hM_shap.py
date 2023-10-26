# https://github.com/shap/shap
# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html#shap.DeepExplainer.explain_row

import os, subprocess
import shap
import numpy as np
import matplotlib.pyplot as plt

import programIO as pIO
import plot as plot
import mlModel as ml


def cmdLineOutput(command, path='./'):
    try:
        return str(subprocess.check_output(command, shell=True, cwd=path)).replace('b\'', '').replace('\'', '')[:-2]
    except:
        return ''
    
def hM(model, input_images, folderLoc):
# # Option_1:
#     # Create a baseline dataset (similar in shape to input_images)
#     baseline_dataset = np.zeros_like(input_images)

#     # ML model evaluation on the best channel configuration
#     Cf, St = ml.ML_prediction(baseline_dataset)
#     # print(Cf.shape)
    
#     # Create a DeepExplainer
#     explainer = shap.DeepExplainer(model, baseline_dataset)


# Option_2:
    # Load and preprocess your input images
    # datasetPath = "/home/ws/vo8312/Desktop/SPP/Alexs_Data/Uploads/ML/ChemEngAI_02"
    datasetPath = cmdLineOutput('git rev-parse --show-toplevel')
    with open(os.path.join(datasetPath, 'X_train_1000.npy'), 'br') as f:
        x_train = np.load(f)

    # select a set of background examples to take an expectation over
    print("1) Training of the background data")
    background = x_train[np.random.choice(x_train.shape[0], 10, replace=False)]

    # ML model evaluation on the best channel configuration
    Cf, St = ml.ML_prediction(background)
    print(Cf.shape)
    
    # Create a DeepExplainer
    print("2) Beginning of SHAP explainer (DeepExplainer)")
    explainer = shap.DeepExplainer(model, background)

    Cf_avg = np.average(Cf)
    St_avg = np.average(St)

    # Generate SHAP values
    print("3) Generating SHAP values")
    shap_values = explainer.shap_values(input_images)
    
    # print("type(shap_values) = ", type(shap_values))  # <class 'list'> , len(shap_values) =  2

    # print("type(shap_values[0]) = ", type(shap_values[0]))  #  <class 'numpy.ndarray'>
    # print("shap_values[0].shape = ", shap_values[0].shape)  # Cf   (numOfinputImages, 129, 384, 1)
    # print("shap_values[1].shape = ", shap_values[1].shape)  # St   (numOfinputImages, 129, 384, 1)
    
    # print("shap_values[0][:,0,0,0].shape = ", shap_values[0][:,0,0,0].shape)  # (2,)
    sM_Cf = []
    sM_St = []
    
    numOfinputImages = shap_values[0][:,0,0,0].shape[0]
    for Imag in range(numOfinputImages):
        binary_mask = input_images[Imag,:,:,0]
        shap_mask_Cf = shap_values[0][Imag,:,:,0]
        shap_mask_St = shap_values[1][Imag,:,:,0]
        
        shap_mask_Cf_sum = np.sum(shap_mask_Cf)
        shap_mask_St_sum = np.sum(shap_mask_St)
        # print("shap_mask_Cf_sum, shap_mask_St_sum = ", round(shap_mask_Cf_sum,4), round(shap_mask_St_sum,4))

        sM_Cf.append(shap_mask_Cf_sum)
        sM_St.append(shap_mask_St_sum)
        # print("Imag, sM_Cf, sM_St = ", Imag, sM_Cf, sM_St)

        # print("Image number = ", Imag)
        # print("shap_mask_Cf.shape = ", shap_mask_Cf.shape)
        # print("shap_mask_Cf.shape = ", shap_mask_Cf.shape)
        
        # plot.shapMask(binary_mask, shap_mask_Cf, folderLoc, Imag, "Cf")
        # plot.shapMask(binary_mask, shap_mask_St, folderLoc, Imag, "St")
        # plot.shapMaskDiff(binary_mask, shap_mask_Cf-shap_mask_St, folderLoc, Imag, "CfmSt")
        
    return sM_Cf, sM_St, Cf_avg, St_avg
    
    
    
     