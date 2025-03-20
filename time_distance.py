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

def plot_jz(mhd_config, tframe, show_plot=False):
    """Plot the z-component of the current density
    """
    run_name = mhd_config["run_name"]
    run_dir = mhd_config["run_dir"]
    output_type = "reconnection.prim"
    
    list_t = []
    list_d = []
    
    for tempo in range (tmin,tmax+1):
        print(tempo)
        fname = run_dir + output_type + "." + str(tempo).zfill(5) + ".athdf"
        fdata = athena_read.athdf(fname)

        time = fdata['Time']
        y = fdata["x2f"]
        rho = fdata['rho'][0]

        ## Density along the current sheet ##
        nxl = len(fdata["x1f"]) // 2
        list_d.append(rho[:,nxl])

    density = np.array(list_d)
    print(density.shape)

    ymin = mhd_config["mesh"]["x2min"]
    ymax = mhd_config["mesh"]["x2max"]
    sizes = [tmin, tmax, ymin, ymax]

    fig = plt.figure(figsize=[9.0, 6.0])
    gs = gridspec.GridSpec(1,
                           3,
                           wspace=0,
                           hspace=0,
                           top=0.95,
                           bottom=0.15,
                           left=0.2,
                           right=0.8)
    rect = [0.11, 0.12, 0.65, 0.8]

    ax = fig.add_axes(rect)
    img = ax.imshow(density.T, cmap=plt.cm.cividis, extent=sizes, 
                    vmin=0.5, vmax=3.0, interpolation='nearest',
                    aspect='auto', origin='lower')
    ax.set_xlabel('$t_A$',fontsize=20)
    ax.set_ylabel('$y$',fontsize=20)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.tick_params(labelsize=15, direction='in')

    rect[0] += rect[2] + 0.02
    rect[2] = 0.03
    cbar_ax = fig.add_axes(rect)
    cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(r'$\rho$',fontsize=15)

    if not show_plot:
        plt.close()

mhd_run_name = "test_4096_2048"
mhd_run_dir = "/anvil/scratch/x-gmurtas/athena_reconnection/test_4096_2048/"
mhd_config = get_mhd_config(mhd_run_dir, "athinput.reconnection")
mhd_config["run_name"] = mhd_run_name
mhd_config["run_dir"] = mhd_run_dir
tmin = 20
tmax = 100

plot_jz(mhd_config, 10, show_plot=True)
plt.savefig("time_distance_1.jpg", dpi=300)
plt.close()

