# athena-reconnection_decks
This repository contains generator files for problems involving reconnecting current sheets run with Athena++ on Purdue ANVIL

Here you can find two problem generators:
- `reconnection.cpp`
- `reconnection_shear.cpp`

The first deck (`reconnection.cpp`) runs a simple 2D double-periodic system with two current sheets, introduced by [Xiaocan Li](https://github.com/xiaocanli). For more information, check https://github.com/xiaocanli/athena_reconnection. The second deck (`reconnection_shear.cpp`) includes a shear velocity profile whose direction is parallel with the $B_{y}$ component of the magnetic field. Once Athena++ is installed, the problem generators can be put in `athena\src\pgen`.
