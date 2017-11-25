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

EXP_NAME = "rnn.scheduled_sampling"
# Run the job
srun python -m train -data data/stackexchange/stackexchange.train_valid.pt -vocab data/stackexchange/stackexchange.vocab.pt -exp_path "exp/$EXP_NAME/%s.uni-directional.%s" -save_path "model/$EXP_NAME/%s.uni-directional.%s" -exp "stackexchange" -batch_size 512 -run_valid_every 1000 -scheduled_sampling_batches 30000
