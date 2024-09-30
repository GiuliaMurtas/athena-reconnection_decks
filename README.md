# athena-reconnection_decks
This repository contains generator files for problems of magnetic reconnection run with Athena++ on Purdue ANVIL.

Here you can find two problem generators:
- `reconnection.cpp`
- `reconnection_shear.cpp`

The first deck (`reconnection.cpp`) runs a simple 2D double-periodic system with two current sheets, introduced by [Xiaocan Li](https://github.com/xiaocanli). For more information, check https://github.com/xiaocanli/athena_reconnection. The second deck (`reconnection_shear.cpp`) includes a shear velocity profile whose direction is parallel with the $B_{y}$ component of the magnetic field. Once Athena++ is installed, the problem generators must be put in `athena\src\pgen`.

The script `plot.py` allow to create a series of 1D and 2D plots for both decks. To work on ANVIL, it requires a Conda environment: please check the documentation at https://www.rcac.purdue.edu/knowledge/anvil/run/examples/apps/python/packages on how to create and load the required Conda environment. By storing the Conda environment in `$HOME/privatemodules`, at every new session load Python via `module load python` and follow with the commands:

`$ module use $HOME/privatemodules` 

`$ module load conda-env/mypackages`

The analysis routine can be then launched by the command `python plot.py`. In order to crate an animation, it is possible to use the `ffmpeg` module, once the plots have been produced by `plot.py`. An example for the current density $J_z$
