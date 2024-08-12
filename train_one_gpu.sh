#!/bin/bash
#SBATCH --account=def-s255khan
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=logs/slurm-%j.out

#set up virtual environment
cd projects/def-s255khan/chaunm/MedSAM
module load StdEnv/2023 python/3.11 gcc/12.3 opencv/4.9.0 scipy-stack/2024a
virtualenv --no-download $SLURM_TMPDIR/medsam-venv
source $SLURM_TMPDIR/medsam-venv/bin/activate
pip install --no-index --upgrade pip

#download dependencies
pip install --no-index torch torchvision scikit-learn albumentations tqdm 
pip install --no-index -e .

python train_one_gpu.py