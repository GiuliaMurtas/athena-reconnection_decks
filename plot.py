import argparse
import errno
import os

import sys
sys.path.insert(0, '/anvil/scratch/x-gmurtas/athena/vis/python')
import athena_read

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.animation as manimation

import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

import math
import json

import h5py
import matplotlib.patches as patches
from scipy.constants import physical_constants
from scipy.interpolate import interp1d
#from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.optimize import curve_fit
from matplotlib import gridspec

plt.rcParams['figure.dpi'] = 300

def mkdir_p(path):
    """Create directory recursively
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_mhd_config(mhd_run_dir, config_name):
    """Get MHD run information from the configuration file

    Arguments:
        mhd_run_dir (string): MHD run directory
        config_name (string): MHD simulation configuration file name
    """
    with open(mhd_run_dir + "/" + config_name) as f:
        contents = f.readlines()
    f.close()
    mhd_config = {}
    for line in contents:
        if "<" in line and ">" in line and "<" == line[0]:
            block_name = line[1:line.find(">")]
            mhd_config[block_name] = {}
        else:
            if line[0] != "#" and "=" in line:
                line_splits = line.split("=")
                tail = line_splits[1].split("\n")
                data = tail[0].split("#")
                ltmp = line_splits[0].strip()
                try:
                    mhd_config[block_name][ltmp] = float(data[0])
                except ValueError:
                    mhd_config[block_name][ltmp] = data[0].strip()
    return mhd_config

#def plot_jz(mhd_config, mhd_config1, tframe, show_plot=False): # For the case at multiple simulations
def plot_jz(mhd_config, tframe, show_plot=False):
    """Plot the z-component of the current density
    """
    run_name = mhd_config["run_name"]
    run_dir = mhd_config["run_dir"]

    #run_name1 = mhd_config1["run_name"]
    #run_dir1 = mhd_config1["run_dir"]

    output_type = "reconnection.prim"
    fname = run_dir + output_type + "." + str(tframe).zfill(5) + ".athdf"
    fdata = athena_read.athdf(fname)

    #fname1 = run_dir1 + output_type + "." + str(tframe).zfill(5) + ".athdf"
    #fdata1 = athena_read.athdf(fname1)

    time = fdata['Time']
    x = fdata["x1f"]
    y = fdata["x2f"]
    dxm = fdata["x1f"][1] - fdata["x1f"][0]
    dym = fdata["x2f"][1] - fdata["x2f"][0]
    bx = fdata['Bcc1'][0]
    by = fdata['Bcc2'][0]
    bz = fdata['Bcc3'][0]
    vx = fdata['vel1'][0]
    vy = fdata['vel2'][0]
    vz = fdata['vel3'][0]
    rho = fdata['rho'][0]
    press = fdata['press'][0]
    jz = np.gradient(by, dxm, axis=1) - np.gradient(bx, dym, axis=0)

    print(time)

    #dxm1 = fdata1["x1f"][1] - fdata1["x1f"][0]
    #dym1 = fdata1["x2f"][1] - fdata1["x2f"][0]
    #bx1 = fdata1['Bcc1'][0]
    #by1 = fdata1['Bcc2'][0]
    #bz1 = fdata1['Bcc3'][0]
    #jz1 = np.gradient(by1, dxm1, axis=1) - np.gradient(bx1, dym1, axis=0)

    #print(np.any(np.isnan(vz)))
    #print(np.any(np.isnan(bz)))

    ## 1D PLOT - Magnetic field across the current sheet ##

    #x = fdata["x1f"]
    #nyl = len(fdata["x2f"]) // 2
    #plt.plot(x[1:], by[nyl,:])
    #plt.ylabel(r'B$_y$',fontsize=12)
    #plt.xlabel(r'x',fontsize=12)
    #plt.title('t = '+str(time),fontsize=15)
    #plt.xlim(0, 4)
    #plt.ylim(-1.05,1.05)

    ## 1D PLOT - Density across the current sheet ##

    #x = fdata["x1f"]
    #nyl = len(fdata["x2f"]) // 2
    #plt.plot(x[1:], rho[nyl,:],label="Slice along x-axis")
    #plt.ylabel(r'$\rho$',fontsize=12)
    #plt.title('t = '+str(round(time,2))+r' t$_{A}$',fontsize=15)
    #plt.xlim(0, 4)
    #plt.savefig("density.jpg")
    #maxima,_ = find_peaks(rho[nyl,:], height=0.7, threshold=None, distance=None)
    #print(maxima,rho[nyl,maxima])
    #plt.scatter(x[maxima],rho[nyl,maxima],s=20,c='red',marker='X')

    ## 1D PLOT - Density along the current sheet ##

    #nxl = len(fdata["x1f"]) // 2
    #plt.plot(rho[:,nxl],y[1:],label="Slice along y-axis")
    #plt.xlabel('x',fontsize=20)
    #plt.ylabel('y',fontsize=20)
    #maxima2,_ = find_peaks(rho[:,nxl], height=0.8, threshold=None, distance=None)
    #print(maxima2,rho[maxima2,nxl])
    #plt.scatter(y[maxima2],rho[maxima2,nxl],s=20,c='red',marker='X')
    #plt.legend(loc="lower right",prop={'size': 10},ncol =1)
    #plt.savefig("density_1D.jpg")

    ## 1D PLOT - Pressure across the current sheet ##

    #x = fdata["x1f"]
    #nyl = len(fdata["x2f"]) // 2
    #plt.plot(x[1:], press[nyl,:],label="Slice along x-axis")
    #plt.ylabel(r'Pressure',fontsize=12)
    #plt.title('t = '+str(round(time,2))+r' t$_{A}$',fontsize=15)
    #plt.xlim(0, 4)
    #plt.savefig("pressure.jpg")
    #plt.xlim(-0.2, 0.2)
    #maxima,_ = find_peaks(rho[nyl,:], height=0.7, threshold=None, distance=None)
    #print(maxima,rho[nyl,maxima])
    #plt.scatter(x[maxima],rho[nyl,maxima],s=20,c='red',marker='X')

    ## 1D PLOT - Inflow velocity across the current sheet ##

    #x = fdata["x1f"]
    #nyl = len(fdata["x2f"]) // 2
    #plt.plot(x[1:], vx[nyl,:],label="Slice along x-axis")
    #plt.ylabel(r'v$_x$',fontsize=12)
    #plt.title('t = '+str(round(time,2))+r' t$_{A}$',fontsize=15)
    #plt.xlim(0, 4)
    #plt.savefig("inflow_velocity.jpg")
    #plt.xlim(-0.2, 0.2)
    #maxima,_ = find_peaks(rho[nyl,:], height=0.7, threshold=None, distance=None)
    #print(maxima,rho[nyl,maxima])
    #plt.scatter(x[maxima],rho[nyl,maxima],s=20,c='red',marker='X')

    ## 1D PLOT - Outflow velocity along the current sheet ##

    #nxl = len(fdata["x1f"]) // 2
    #plt.plot(y[1:],vy[:,nxl],label="Slice along y-axis")
    #maxima2,_ = find_peaks(vy[:,nxl], height=0.8, threshold=None, distance=None)
    #print(maxima2,vy[maxima2,nxl])
    #plt.scatter(y[maxima2],vy[maxima2,nxl],s=20,c='red',marker='X')
    #plt.legend(loc="lower right",prop={'size': 10},ncol =1)
    #plt.savefig("outflow_velocity_1D.jpg")

    ## Compression rate ##

    #ro_rate = rho[nyl,maxima[0]]/rho[maxima2[0],nxl]
    #print(r'Compression rate =',ro_rate)

    ## 1D PLOT - Current density across the current sheet ##

    #x = fdata["x1f"]
    #nyl = len(fdata["x2f"]) // 2
    #plt.plot(x[1:], jz[nyl,:])

    #x1 = fdata1["x1f"]
    #nyl1 = len(fdata1["x2f"]) // 2
    #plt.plot(x1[1:], jz1[nyl1,:])

    #plt.ylabel(r'J$_z$',fontsize=12)
    #plt.xlabel(r'x',fontsize=12)
    #plt.title('t = '+str(time),fontsize=15)
    #plt.xlim(1.9, 2.1)
    #ticks = np.arange(1.9,2.1,0.05)
    #tickla = [f'{tick:1.2f}' for tick in ticks]
    #plt.xticks(ticks, tickla)
    #plt.show()

    ## Calculation of the FWHM ##

    #d = jz[nyl,:] - (max(jz[nyl,:]) / 2.)
    #indexes = np.where(d > 0)[0]
    #print(abs(x[indexes[-1]] - x[indexes[0]]))

    ## 2D plot of the current density magnitude 

    xmin = mhd_config["mesh"]["x1min"]
    xmax = mhd_config["mesh"]["x1max"]
    ymin = mhd_config["mesh"]["x2min"]
    ymax = mhd_config["mesh"]["x2max"]
    sizes = [xmin, xmax, ymin, ymax] #[xmin+1.5, xmax-1.5, ymin, ymax] for 4x2 domain

    fig = plt.figure(figsize=[9.5, 8.5])
    gs = gridspec.GridSpec(1,
                           3,
                           wspace=0,
                           hspace=0,
                           top=0.95,
                           bottom=0.15,
                           left=0.2,
                           right=0.8)
    rect = [0.11, 0.12, 0.65, 0.8]

    dt_out = mhd_config["output1"]["dt"]
    tva = dt_out * tframe
    #tva = dt_out * tframe / (ymax - ymin)
    
    #print(r'vz:',np.max(vz),np.min(vz))
    #print(r'vx:',np.max(vx),np.min(vx))
    #print(r'vy:',np.max(vy),np.min(vy))
    
    print(r'Density:',np.max(rho),np.min(rho))
    print(r'x-velocity:',np.max(vx),np.min(vx))

    #print(r'Maximum pressure:',np.max(press))
    #print(r'Maximum density:',np.max(rho))
    #print(r'Maximum temperature:',np.max(press/rho))
    #print(r'Minimum temperature:',np.min(press/rho))


    ax = fig.add_axes(rect)
    img = ax.imshow(jz,cmap=plt.cm.seismic, extent=sizes,  #jz[0:4096,3072:5120] for 4x2 domain #cividis for density, seismic for J
                    vmin=-100, vmax=100, #0.5 to 1.5 for density, -100 to 100 for current density
                    aspect='equal', origin='lower')
    ax.set_title('t = '+str(round(tva,2))+r' $\tau_{A}$',fontsize=20)
    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('y',fontsize=20)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.set_xticklabels(['-0.5', '0.0', '0.5'])
    ax.tick_params(labelsize=15, direction='in')
    strm = ax.streamplot(x[1:],y[1:],bx,by,linewidth=0.5,color='k',density=1.0,arrowstyle='-')
    
    rect[0] += rect[2] + 0.02
    rect[2] = 0.03
    cbar_ax = fig.add_axes(rect)
    cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(r'$J_{z}$',fontsize=15)

    if not show_plot:
        plt.close()

mhd_run_name = "test_4096_2048"
mhd_run_dir = "/anvil/scratch/x-gmurtas/athena_reconnection/test_4096_2048/"
mhd_config = get_mhd_config(mhd_run_dir, "athinput.reconnection")
mhd_config["run_name"] = mhd_run_name
mhd_config["run_dir"] = mhd_run_dir

#mhd_run_name1 = "reconnection_test_15"
#mhd_run_dir1 = "/global/cscratch1/sd/gmurtas/athena_reconnection_test/reconnection_test_15/"
#mhd_config1 = get_mhd_config(mhd_run_dir1, "athinput.reconnection")
#mhd_config1["run_name"] = mhd_run_name1
#mhd_config1["run_dir"] = mhd_run_dir1

#for x in range (0,10):
    #print,x
    #plot_jz(mhd_config, mhd_config1, x, show_plot=True)
#    plot_jz(mhd_config, x, show_plot=True)
#    plt.savefig("Jz_00"+str(x)+".jpg", dpi=300)
#    plt.close()

#for x in range (10,100):
#    print,x
#    plot_jz(mhd_config, x, show_plot=True)
#    plt.savefig("Jz_0"+str(x)+".jpg", dpi=300)
#    plt.close()

for x in range (100,201):
#    print,x
    plot_jz(mhd_config, x, show_plot=True)
    plt.savefig("Jz_"+str(x)+".jpg", dpi=300)
    plt.close()
