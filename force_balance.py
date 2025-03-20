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
from scipy.optimize import curve_fit
from matplotlib import gridspec

from sympy import symbols
from sympy.physics.vector import ReferenceFrame, gradient, divergence, curl

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
    
def cross_prod(a, b):
    result = [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]
    return result

def plot_jz(mhd_config, tframe, show_plot=False):
    """Plot the z-component of the current density
    """
    run_name = mhd_config["run_name"]
    run_dir = mhd_config["run_dir"]

    output_type = "reconnection_2.prim"
    fname = run_dir + output_type + "." + str(tframe).zfill(5) + ".athdf"
    fdata = athena_read.athdf(fname)

    time = fdata['Time']
    x = fdata["x1f"]
    y = fdata["x2f"]
    z = fdata["x3f"]
    dxm = fdata["x1f"][1] - fdata["x1f"][0]
    dym = fdata["x2f"][1] - fdata["x2f"][0]
    dzm = fdata["x3f"][1] - fdata["x3f"][0]
    bx = fdata['Bcc1'][0]
    by = fdata['Bcc2'][0]
    bz = fdata['Bcc3'][0]
    vx = fdata['vel1'][0]
    vy = fdata['vel2'][0]
    vz = fdata['vel3'][0]
    rho = fdata['rho'][0]
    press = fdata['press'][0]

    dummy, dbx_dy = np.gradient (bx, dxm, dym, axis=[0,1])
    dby_dx, dummy = np.gradient (by, dxm, dym, axis=[0,1])
    dbz_dx, dbz_dy= np.gradient (bz, dxm, dym, axis=[0,1])

    jx = dbz_dy
    jy = - dbz_dx
    jz = dby_dx - dbx_dy
    
    ## Magnetic part ##
    b = np.array([bz,by,bx])
    j = np.array([jz,jy,jx])
    jcrossb = cross_prod(j,b)
    col_vec = np.array(jcrossb, ndmin=3)
    print(r'Shape of B (z,y,x):',np.shape(b))
    print(r'Shape of J (z,x,y):',np.shape(j))
    print(r'Shape of JxB (z,y,x):',np.shape(jcrossb))
    result=np.squeeze(jcrossb[2:])
    print(r'New shape of JxB:',result.shape)
    
    ## Gradient of gas pressure ##
    grad_p = np.gradient(press)
    print(r'Shape of pressure gradient (y,x):',np.shape(press))
    print(r'Maximum grad p:',np.max(grad_p),r'Minimum grad p:',np.min(grad_p))
    
    ## Gradient of magnetic pressure ##
    #grad_B = 0.5*np.gradient(np.power(bx,2)+np.power(by,2)+np.power(bz,2))
    
    ## Force Balance ##
    force = np.subtract(result,grad_p)
    print(r'Shape of force balance elements:',force.shape)
    force_tot=np.squeeze(force[1:])

    print(r'Time output = ',time)
    print(r'Maximum force:',np.max(force_tot),r'Minimum force:',np.min(force_tot))

    xmin = mhd_config["mesh"]["x1min"]
    xmax = mhd_config["mesh"]["x1max"]
    ymin = mhd_config["mesh"]["x2min"]
    ymax = mhd_config["mesh"]["x2max"]
    sizes = [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=[5.5, 7.5])
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

    ax = fig.add_axes(rect)
    img = ax.imshow(force_tot,cmap=plt.cm.seismic, extent=sizes,
                    vmin=-1, vmax=1,
                    aspect='equal', origin='lower')
    ax.set_title('t = '+str(round(tva,2))+r' $\tau_{A}$',fontsize=20)
    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('y',fontsize=20)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.tick_params(labelsize=15, direction='in')

    rect[0] += rect[2] + 0.02
    rect[2] = 0.03
    cbar_ax = fig.add_axes(rect)
    cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(r'$J \times B - \nabla p$',fontsize=15)

    if not show_plot:
        plt.close()

mhd_run_name = "beta_0.16_ff_3"
mhd_run_dir = "/anvil/scratch/x-gmurtas/athena_reconnection/beta_0.16_ff_3/"
mhd_config = get_mhd_config(mhd_run_dir, "athinput.reconnection")
mhd_config["run_name"] = mhd_run_name
mhd_config["run_dir"] = mhd_run_dir

for x in range (0,10):
    plot_jz(mhd_config, x, show_plot=True)
    plt.savefig("force_00"+str(x)+".jpg", dpi=300)
    plt.close()

for x in range (10,100):
    plot_jz(mhd_config, x, show_plot=True)
    plt.savefig("force_0"+str(x)+".jpg", dpi=300)
    plt.close()

#for x in range (100,201):
#    plot_jz(mhd_config, x, show_plot=True)
#    plt.savefig("Jz_"+str(x)+".jpg", dpi=300)
#    plt.close()
