#!/bin/bash
#SBATCH -A p30771
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -o /home/jma819/miniscope_denoising/miniscope_v4_preprocessing/logfiles/slurm.%x-%j.out # STDOUT
#SBATCH --job-name="slurm_v4_preprocessing"
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=20G


module purge all
cd ~
#add project directory to PATH
export PATH=$PATH/projects/p30771/
export PATH=$PATH/scratch/jma819/
export PATH=$PATH/projects/b1118

#get inputs from command line and run

INPUT_behavCamDirectory = $1
INPUT_regExp = $2


echo "running conversion, contrast enhancement"

module load python/anaconda3
source activate image_processing_env

python batchContrastAdjustScript.py $INPUT_behavCamDirectory $INPUT_regExp