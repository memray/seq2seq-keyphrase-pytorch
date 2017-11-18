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
#srun python -m train -data data/stackexchange/stackexchange.train_valid.pt -vocab data/stackexchange/stackexchange.vocab.pt -exp_path "exp/stackexchange.uni-directional.no-loss-mask.%s" -exp "stackexchange" -batch_size 256
srun python -m train -data data/stackexchange/stackexchange.train_valid.pt -vocab data/stackexchange/stackexchange.vocab.pt -exp_path "exp/stackexchange.uni-directional.no-loss-mask.20171117-214930/" -exp "stackexchange" -batch_size 256 -train_from "exp/stackexchange.uni-directional.no-loss-mask.20171117-214930/stackexchange.epoch=19.batch=276.total_batch=42000.model"
