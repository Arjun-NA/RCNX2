#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=arjuna@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=2-0:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=REsCap
#SBATCH --output=test_%j.txt


######  Module commands #####



######  Job commands go below this line #####
source activate rescapnet
cd /N/slate/arjuna/Software_Residual_Capsule_Network
python RCN2.py --testing --weight ./result/85perc/trained_model.h5
