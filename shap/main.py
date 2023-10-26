

import os
import time
import subprocess

import mlModel as ml
import generateGeo as gG
import geoParam as gP
import plot as plot
import programIO as pIO
import hM_shap as hM

def cmdLineOutput(command, path='./'):
    try:
        return str(subprocess.check_output(command, shell=True, cwd=path)).replace('b\'', '').replace('\'', '')[:-2]
    except:
        return ''
    
def main(model, folderLoc, bT, bH):
    
    numOfBlocks = 100
    x1, delta = gG.generate_x1(numOfBlocks, folderLoc, bT, bH)
    
    # ML model evaluation on the best channel configuration
    Cf, St = ml.ML_prediction(x1)
    
    print("type(delta) = ", type(delta))
    print("type(Cf) = ", type(Cf))
    print("Cf.shape, St.shape, delta.shape = ", Cf.shape, St.shape, delta.shape)
    print("Cf, St, solidFraction = ", Cf/0.06, St/0.016, 1-delta)
    
    pIO.writeToFile(Cf, St, delta, folderLoc)
    plot.plotCfSt(Cf, St, delta, folderLoc, "CfSt")
    plot.plotCfSt(Cf/0.06, St/0.016, delta, folderLoc, "CfSt_normlised")

    Cf_xtrainAvg = 0.07148285    # average value of Cf for 8726 training data
    St_xtrainAvg = 0.016369773   # average value of St for 8726 training data
    
    #Heat Map calculation using SHAP
    Cf_shapSum, St_shapSum, Cf_backgroundAvg, St_backgroundAvg = hM.hM(model, x1, folderLoc)
    pIO.writeToFileShap(Cf, St, delta, 
                        Cf_shapSum, St_shapSum, 
                        Cf_backgroundAvg, St_backgroundAvg, 
                        Cf_xtrainAvg, St_xtrainAvg, folderLoc)

    
    
if __name__ == "__main__":
    
    # Loading the ML model 
    model, y_scaler = ml.loadMLmodel()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #to disable GPU with TensorFlow (if you want to run it on CPUs) 
    gitRepo = cmdLineOutput('git rev-parse --show-toplevel')
    
    if gP.flag_arragment==0: #staggerd
        flag = "Stag"
    else:
        flag = "NStag"

    T1 = time.perf_counter()

    blockHeight = gP.blockHeight
    blockThickness = gP.blockThickness
    
    for bH in range(len(blockHeight)):
        for bT in range(len(blockThickness)):
            dataFolder = f"figures_bT_{blockThickness[bT]}_bH_{blockHeight[bH]}_{flag}"
            folderLoc = os.path.join(gitRepo, dataFolder)
                    
            os.system(f'rm -r {folderLoc}')
            os.system(f'mkdir -p {folderLoc}')
            
            main(model, folderLoc, bT, bH)
        
    T2 = time.perf_counter()
    print(f'The main code finished in {round(T2-T1, 6)} second(s)')
    
