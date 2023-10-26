import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import geoParam as gP


def plotSettings(legendLocation, xlabel, ylabel):
    # creating the font properties
    font1 = {'family':'Arial','color':'black'}
    
    ax = plt.gca() #you first need to get the axis handle
    ax.minorticks_on()
    ax.tick_params(which='minor', width=0.1)  
    # ax.set_aspect('auto') 

    plt.ylabel(ylabel)
    plt.xlabel(xlabel, fontdict=font1, fontsize=15, labelpad=10)
    plt.grid(linestyle='dashed', linewidth=0.3)
    plt.grid(linestyle='dotted',  which='minor', linewidth=0.16)
  
    formatter = tick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-2,2)) 
    ax.yaxis.set_major_formatter(formatter) 
    
    
def plotCfSt(variable_1, variable_2, variable_3, destDir_plot, figname):
    fig, ax = plt.subplots(figsize=(5,2))
    # plt.plot(range(len(variable_1)), variable_1, marker="o", ms=1, alpha=0.5, color="black", label = "Cf")
    # plt.plot(range(len(variable_2)), variable_2, marker="x", ms=1, alpha=0.5, color="red", label = "St")
    plt.plot((1.0 - variable_3), variable_1, marker="o", ms=1, alpha=0.5, color="black", label = "Cf")
    plt.plot((1-0 - variable_3), variable_2, marker="x", ms=1, alpha=0.5, color="red", label = "St")
    plotSettings("upper right", 'solid fraction', '')
    plt.legend()
    plt.savefig(f'{destDir_plot}/{figname}.png', format='png', dpi=300, bbox_inches='tight')
    # plt.show()  
    plt.close()

    

def writeSingleMask(mask, iter, output_dir):
    print(f"Writing mask for iter {iter}")
    fig, ax = plt.subplots(figsize=(10,2))
    c = ax.pcolormesh(gP.X, gP.Y, mask)
    fig.colorbar(c, ax=ax)
    plt.savefig(f'{output_dir}/mask_{iter}.png', format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def shapMask(bmask, smask, foldloc, ImagNum, name):    
    plt.figure(figsize=(10,2), frameon=False) 
    
    # vmin = round(smask.min(),2)
    # vmax = round(smask.max(),2)
    # print("vmin, vmax = ", vmin, vmax)
    # norm = colors.TwoSlopeNorm(vmin=vmin/20, vcenter=0, vmax=vmax/20)

    # vmin, vmax = -0.004, 0.004
    vmin, vmax = -0.01, 0.01
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    contourf_shap = plt.pcolor(gP.mesh_x, gP.mesh_y, smask, norm=norm, cmap = 'bwr', rasterized=True)
    contourf_ibm = plt.contour(gP.mesh_x, gP.mesh_y, bmask, [0.5], colors=('k',), linewidths=0.75, alpha=0.80)
    
    ax = plt.gca() #you first need to get the axis handle
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.20)
    cbar = plt.colorbar(contourf_shap, cax = cax)
    cbar.set_label("shap values", rotation=90, labelpad= 5)
    cbar.ax.locator_params(nbins=5)
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.4f'))
    
    plt.tick_params(bottom=True, top=True, left=True, right=True, colors = 'w', labelcolor = 'k', direction = 'in')

    plt.savefig(f'{foldloc}/shapMask_{ImagNum}_{name}.png', format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()



def shapMaskDiff(bmask, smask, foldloc, ImagNum, name):    
    plt.figure(figsize=(10,2), frameon=False) 
    
    vmin = round(smask.min(),2)
    vmax = round(smask.max(),2)
    print("vmin, vmax = ", vmin, vmax)
    maximum = np.max([np.abs(vmin), vmax])
    print("maximum = ", maximum)
    if maximum < 0.1:
        vmax = 0.004
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = colors.TwoSlopeNorm(vmin=-maximum/20, vcenter=0, vmax=maximum/20)
    
    contourf_shap = plt.pcolor(gP.mesh_x, gP.mesh_y, smask, norm=norm, cmap = 'bwr', rasterized=True)
    contourf_ibm = plt.contour(gP.mesh_x, gP.mesh_y, bmask, [0.5], colors=('k',), linewidths=0.75, alpha=0.80)
    
    # ax = plt.gca() #you first need to get the axis handle
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.20)
    # cbar = plt.colorbar(contourf_shap, cax = cax)
    # cbar.set_label("shap values", rotation=90, labelpad= 5)
    # cbar.ax.locator_params(nbins=5)
    # cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.4f'))
    
    plt.tick_params(bottom=True, top=True, left=True, right=True, colors = 'w', labelcolor = 'k', direction = 'in')

    plt.savefig(f'{foldloc}/shapMask_{ImagNum}_{name}.png', format='png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

