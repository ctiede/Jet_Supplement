#!/bin/bash
#PBS -l nodes=2:ppn=8
#PBS -N IG

cd $PBS_O_WORKDIR


Input=~/Data/
Output=~/Image/

mpiexec ./video_parallel video_pyjet_3d.py   ${Input} ${Output}
