#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=arjuna@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2-0:00:00
#SBATCH --mem=600gb
#SBATCH --partition=gpu
#SBATCH --gpus v100:4
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=RESCAP
#SBATCH --output=Hparam/hparam_tuner_%j.txt


######  Module commands #####
module load anaconda


######  Job commands go below this line #####
source activate rescapnet
cd /N/slate/arjuna/RCNX2/
nnictl create --config config.yml
nvidia-smi -l 20
