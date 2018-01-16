#!/usr/bin/env bash

#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=pack_padded_sequence.attn_concat.rnn.teacher_forcing.copy
#SBATCH --output=slurm_output/pack_padded_sequence.attn_concat.rnn.teacher_forcing.copy.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB

# Load modules
#module restore

# Run the job
export ATTENTION="concat";
export EXP_NAME="pack_padded_sequence.attn_$ATTENTION.rnn.teacher_forcing.copy"
export DATA_NAME="kp20k"
srun python -m train -data data/$DATA_NAME/$DATA_NAME -vocab data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/exp/$EXP_NAME/%s.%s" -model_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/model/$EXP_NAME/%s.%s" -pred_path "/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/pred/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size 256 -bidirectional -copy_model -run_valid_every 2000 -save_model_every 10000 -must_teacher_forcing -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode $ATTENTION
