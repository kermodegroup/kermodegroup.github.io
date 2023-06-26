#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3

# Avon
module purge
module load GCC OpenMPI torchvision/0.13.1-CUDA-11.7.0

python3 ./mace/scripts/run_train.py \
    --name="InP_MACE" \
    --train_file="training_structures.xyz" \
    --valid_fraction=0.15 \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='32x0e + 32x1o' \
    --r_max=6.0 \
    --batch_size=50 \
    --max_num_epochs=500 \
    --swa \
    --start_swa=450 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --stress_key="stress"
