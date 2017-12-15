#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=train-kp20k-unidirectional
#SBATCH --output=slurm_output/train-kp20k-unidirectional.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB

# Load modules
#module restore

# Run the job
export EXP_NAME="rnn.teacher_forcing"
export DATA_NAME="kp20k"
srun python -m train -data data/$DATA_NAME/$DATA_NAME -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "exp/$EXP_NAME/%s.%s" -model_path "model/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size 256 -run_valid_every 2000 -save_model_every 2000 -must_teacher_forcing -beam_size 16 -beam_search_batch_size 32