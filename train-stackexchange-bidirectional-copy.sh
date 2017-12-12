#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=train-stackexchange-bidirectional-copy
#SBATCH --output=slurm_output/train-stackexchange-bidirectional-copy.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB

# Load modules
#module restore

# Run the job
export EXP_NAME="rnn.teacher_forcing.copy"
export DATA_NAME="stackexchange"
srun python -m train -data data/$DATA_NAME/$DATA_NAME -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "exp/$EXP_NAME/%s.%s" -save_path "model/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size 256 -bidirectional -copy_model -run_valid_every 1000 -save_model_every 1000 -must_teacher_forcing -beam_size 16 -beam_search_batch_size 16
