#!/usr/bin/zsh
 
#SBATCH --job-name=3DCDC-NAS
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=output.%J.txt
#SBATCH --time=08:00:00
 
export CONDA_ROOT=$HOME/anaconda3 
export PATH="$CONDA_ROOT/bin:$PATH"
 
source activate pytorch222
 
module load CUDA
echo; export; echo; nvidia-smi; echo
 
sh ./run_hpc.sh