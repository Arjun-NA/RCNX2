#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=arjuna@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=2-0:00:00
#SBATCH --mem=128gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=RCNX2
#SBATCH --output=Train/train_%j.txt


######  Module commands #####
module load anaconda


######  Job commands go below this line #####
source activate rescapnet
cd /N/slate/arjuna/RCNX2
python RCN2X.py
