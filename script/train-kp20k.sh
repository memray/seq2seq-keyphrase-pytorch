#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --partition=gtx1080
#SBATCH --job-name=rnn.kp20k.multi_test.general
#SBATCH --output=slurm_output/train.rnn.kp20k.multi_test.general.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore

# Run the job
export EXP_NAME="rnn.kp20k.multi_test.general"
export ATTENTION="general"
export ROOT_PATH="/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch"
export DATA_NAME="kp20k"
srun python -m train -data data/$DATA_NAME/$DATA_NAME -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp "$DATA_NAME" -batch_size 128 -bidirectional -run_valid_every 5000 -save_model_every 5000 -beam_size 32 -beam_search_batch_size 3 -train_ml
