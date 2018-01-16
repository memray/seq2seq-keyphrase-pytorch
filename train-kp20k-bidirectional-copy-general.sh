#!/usr/bin/env bash

export DATA_NAME="kp20k"
export ATTENTION="general"
export EXP_NAME="rnn.teacher_forcing.copy.pack_padded_sequence.attn_$ATTENTION"

echo "DATA_NAME=$DATA_NAME, ATTENTION=$ATTENTION, EXP_NAME=$EXP_NAME";

export ARGUMENT="-data data/$DATA_NAME -vocab data/$DATA_NAME.vocab.pt -exp_path /zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/exp/$EXP_NAME/%s.%s -model_path /zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/model/$EXP_NAME/%s.%s -pred_path /zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch/pred/$EXP_NAME/%s.%s -exp $DATA_NAME -batch_size 128 -bidirectional -copy_model -run_valid_every 2000 -save_model_every 10000 -must_teacher_forcing -beam_size 16 -beam_search_batch_size 32 -train_ml -attention_mode $ATTENTION"

echo $ARGUMENT

sbatch --export=EXP_NAME=$EXP_NAME,ARGUMENT=$ARGUMENT --job-name=$EXP_NAME --output=slurm_log/$EXP_NAME.out train.sbatch;
