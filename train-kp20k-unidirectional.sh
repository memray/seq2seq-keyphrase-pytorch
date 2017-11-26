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
srun EXP_NAME = "rnn.scheduled_sampling"
srun python -m train -data data/kp20k/kp20k.train_valid.pt -vocab data/kp20k/kp20k.vocab.pt -exp_path "exp/$EXP_NAME/%s.uni-directional.%s" -save_path "model/$EXP_NAME/%s.uni-directional.%s" -exp "kp20k" -batch_size 256 -run_valid_every 2000 -scheduled_sampling_batches 60000
