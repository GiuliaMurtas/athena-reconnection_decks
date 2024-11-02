# athena-reconnection_decks
This repository contains generator files for problems of magnetic reconnection run with Athena++ on Purdue ANVIL.

Here you can find three problem generators:
- `reconnection.cpp`
- `reconnection_shear.cpp`
- `reconnection_close.cpp`

The first deck (`reconnection.cpp`) runs a simple 2D double-periodic system with two current sheets, introduced by [Xiaocan Li](https://github.com/xiaocanli). For more information on how to install and run Athena++, check https://github.com/xiaocanli/athena_reconnection. The second deck (`reconnection_shear.cpp`) includes a shear velocity profile whose direction is parallel with the $B_{y}$ component of the magnetic field. The third deck (`reconnection_close.cpp`) presents the same initial conditions as `reconnection.cpp`, but an additional boundary condition (no-slip close boundaries) is available for the $y$-direction. Once Athena++ is installed, the problem generators must be put in `athena\src\pgen`.

## Compile and run a simulation on Purdue ANVIL

In order to run a simulation on ANVIL, the following files need to be copied into the simulation directory:

- `compile_reconnection.sh`
- `athinput.reconnection_openbc`
- `athinput.reconnection_hall`
- `anvil.batch`

Before compiling, a few modules are needed to run these script on Purdue ANVIL:

`> module load parallel openmpi hdf5 python`

The script `compile_reconnection.sh` is used for configuring and compiling the code. To compile the code, run `> chmod u+x compile_reconnection.sh` to make it executable, then use `> ./compile_reconnection.sh`.

## Plot the results

The script `plot.py` allow to create a series of 1D and 2D plots for both decks. To work on ANVIL, it requires a Conda environment: please check the documentation at https://www.rcac.purdue.edu/knowledge/anvil/run/examples/apps/python/packages on how to create and load the required Conda environment. By storing the Conda environment in `$HOME/privatemodules`, at every new session load Python via `> module load python` and follow with the commands:

```
> module use $HOME/privatemodules
> module load conda-env/mypackages
```

The analysis routine requires a few Python libraries in order to work. These can be installed via Conda:

`> conda install numpy matplotlib h5py scipy`

The routine can then be launched with the command `> python plot.py`. In order to create an animation, it is possible to use the `ffmpeg` module, once the plots have been produced by `plot.py`. This module can be simply loaded with `> module load ffmpeg`. An example of the command to produce an animation with `ffmpeg` is:

`> ffmpeg -framerate 10 -i plot_name_%03d.jpg your_file.mp4`
