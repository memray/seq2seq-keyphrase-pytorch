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
export EXP_NAME="rnn.scheduled_sampling"
export DATA_NAME="kp20k"
srun python -m train -data data/$DATA_NAME/$DATA_NAME.train_valid.pt -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/exp/$EXP_NAME/%s.%s" -model_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/model/$EXP_NAME/%s.%s" -pred_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/pred/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size 256 -bidirectional -run_valid_every 2000 -save_model_every 10000 -scheduled_sampling -scheduled_sampling_batches 200000 -train_ml
