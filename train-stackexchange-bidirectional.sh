#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=tdr_dag
#SBATCH --output=tdr_dag.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore

# Run the job
export EXP_NAME="rnn.teacher_forcing"
export DATA_NAME="stackexchange"
srun python -m train -data data/$DATA_NAME/$DATA_NAME.train_valid.pt -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "exp/$EXP_NAME/%s.bi-directional.%s" -save_path "model/$EXP_NAME/%s.bi-directional.%s" -exp "$DATA_NAME" -batch_size 256 -bidirectional -run_valid_every 1000