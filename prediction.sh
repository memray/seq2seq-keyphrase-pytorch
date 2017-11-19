#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=prediction
#SBATCH --output=prediction.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore

# Run the job
directory_exp='/ihome/pbrusilosky/rum20/seq2seq-keyphrase-pytorch/exp/'
directory_data='/ihome/pbrusilosky/rum20/seq2seq-keyphrase-pytorch/data/'

srun python -m predict  -vocab ${directory_data}/kp20k/kp20k.vocab.pt -save_data ${directory_data}/kp20k/kp20k -test_data ${directory_data}/kp20k/kp20k_testing.json -model_path  ${directory_exp}/kp20k.bi-directional.no-loss-mask.20171117-214914/kp20k.epoch=1.batch=8000.total_batch=8000.model -bidirectional
