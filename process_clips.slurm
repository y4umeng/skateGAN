#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.


#SBATCH -J slurm_job #Slurm job name

# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
##SBATCH --mail-user=user_name@ust.hk #Update your email address
##SBATCH --mail-type=begin
##SBATCH --mail-type=end

# Choose partition (queue) "cpu" or "cpu-share" or replace with partition you can access
#SBATCH -p cpu-share

# Use 2 nodes and 80 cores
#SBATCH -N 2 -n 80

# Setup runtime environment if necessary
# For example, setup intel MPI environment
source activate pytorch3d 

# Run your application
python process_clips.py