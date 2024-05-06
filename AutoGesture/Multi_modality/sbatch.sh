#!/usr/bin/zsh
 
#SBATCH --job-name=3DCDC-NAS
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=output.%J.txt
#SBATCH --time=08:00:00
 
export CONDA_ROOT=$HOME/anaconda3 
export PATH="$CONDA_ROOT/bin:$PATH"
 
source activate pytorch222
 
module load CUDA
echo; export; echo; nvidia-smi; echo
 
# 在 tmux 会话中运行 visdom
tmux new-session -d -s visdom 'visdom'

# 等待5秒钟以确保 visdom 启动
sleep 5

# 在 tmux 的同一会话中新开一个窗口执行 sh ./run_hpc.sh
sh ./run_hpc.sh