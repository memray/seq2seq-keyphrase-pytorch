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
srun python -m train -data data/$DATA_NAME/$DATA_NAME.train_valid.pt -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "exp/$EXP_NAME/%s.uni-directional.%s" -save_path "model/$EXP_NAME/%s.uni-directional.%s" -exp "$DATA_NAME" -batch_size 512 -run_valid_every 1000 -teacher_forcing_ratio 1
