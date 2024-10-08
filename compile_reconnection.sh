#!/bin/bash
#-------------------------------------------------------------------------------
# This script is used to run the athena code in the local machine.
# Update:
#   2016-03-21: works on islington.
#   2018-10-21: works on Macbook pro.
#   2018-04-30: Nersc.
#-------------------------------------------------------------------------------

# (0) Initialize
athena_path=$SCRATCH/athena
runs_backup_path=$SCRATCH/mhd_runs_backup
work_path=$PWD
pname=${PWD##*/}
ctime=`date '+%Y%m%d_%H%M%S'`
source_name='source_'${ctime}'.tar'
prob_name=reconnection_shear

# Add library path
module load openmpi
module list 2>&1 | tee modules_list

# (1) Step 1: compile
cd $athena_path
(python configure.py --prob ${prob_name} -b --flux hlld -mpi -omp -hdf5) &&
make clean
make

cd ../
tar -cf ${source_name} athena
mv ${source_name} $work_path
cp $athena_path/bin/athena $work_path
cd $work_path
echo $work_path > mhd_run_dir
cd ..
tar zcvf ${pname}.tar.gz $pname 
mkdir -p $runs_backup_path
mv ${pname}.tar.gz $runs_backup_path
cd $work_path

echo 'Compile: done.'
