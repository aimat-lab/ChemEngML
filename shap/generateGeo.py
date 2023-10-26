

import os
import numpy as np

import geoParam as gP
import programIO as pIO
import plot as plot


def maskInit():
    buffe_index = gP.plateThickness
    index_lower = buffe_index
    index_upper = gP.nyp - buffe_index 

    binary_mask = np.zeros([gP.nxp, gP.nyp]) #initialising
    
    for nx in range(gP.nxp):
        binary_mask[nx, 0:index_lower] = 1 
        binary_mask[nx, index_upper:] = 1 
        
    return binary_mask


def channelHeight(binary_mask):
    #To calculate the channel melt-down height
    bM_x=np.mean(binary_mask,axis=0)
    bM_xy=np.trapz(bM_x,gP.mesh_y)
    
    delta_eff = (2.0 - bM_xy)/2.0  # effective channel half height is equivalent to fluid fraction

    return delta_eff


def generate_x1(numOfBlocks, folderLoc, bT, bH):
    num =0 
        
#Initial channel geometery
    binary_mask = maskInit()
    delta_eff = channelHeight(binary_mask)
    delta_collectn = [delta_eff]

    plot.writeSingleMask(binary_mask.T, num, folderLoc)
    ibm_mask_tpose = binary_mask.T
    x1 = ibm_mask_tpose.reshape((1, gP.nyp, gP.nxp, 1)) #IMPORTANT: the input to the ML model has to be in this shape
    x1_collectn = np.concatenate([x1])

#Adding blocks to channel geometery
    t = gP.blockThickness[bT]//2
    h_l = gP.blockHeight[bH]
    h_u = gP.nyp - gP.blockHeight[bH]
    
    blockThickness_flag = 0
        
    for structure_num in range(0, numOfBlocks+1):
        binary_mask = maskInit()
        
        delta_x = gP.nxp / (structure_num + 1)
        if not (int(delta_x) < int(1.25*gP.blockThickness[bT])):
            x_loc_u = [int(delta_x * i + (delta_x/2)) for i in range(0, structure_num + 1)]
            if gP.flag_arragment==0: #staggerd
                x_loc_l = [int(delta_x * i) for i in range(0, structure_num + 1)]
            else:
                x_loc_l = x_loc_u

            # print(delta_x, x_loc_l,  x_loc_u, gP.mesh_x[x_loc_l])
            # print(x_loc_l[0]-t, x_loc_l[0]+t)

            for i in range(len(x_loc_l)):
                binary_mask[x_loc_l[i]-t:x_loc_l[i]+t, 0:h_l] = 1 
                binary_mask[x_loc_u[i]-t:x_loc_u[i]+t, h_u:] = 1 
            
            # for the first block in the upper wall    
            binary_mask[int(delta_x/2)-t:int(delta_x/2)+t, h_u:] = 1 
            
            if gP.flag_arragment==0: #staggerd
                # for the first block in the lower wall    
                binary_mask[0:t, 0:h_l] = 1 
                binary_mask[gP.nxp-t:, 0:h_l] = 1 
        else:
            print("structure_num, delta_x = ", structure_num, delta_x)
            binary_mask[:, 0:h_l] = 1 
            binary_mask[:, h_u:] = 1 
            
            blockThickness_flag = 1


        delta_eff = channelHeight(binary_mask)

        plot.writeSingleMask(binary_mask.T, structure_num+1, folderLoc)
        
        ibm_mask_tpose = binary_mask.T
        x1 = ibm_mask_tpose.reshape((1, gP.nyp, gP.nxp, 1)) #IMPORTANT: the input to the ML model has to be in this shape
        x1_collectn = np.concatenate([x1_collectn, x1], axis=0)
        
        delta_collectn.append(delta_eff)
        
        if blockThickness_flag == 1:
            break    # break here
        
    return x1_collectn, np.array(delta_collectn)
