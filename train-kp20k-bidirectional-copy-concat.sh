#!/usr/bin/env bash

export DATA_NAME="kp20k";
export ATTENTION="concat";
export EXP_NAME="attn_$ATTENTION.pack_padded_sequence.teacher_forcing.copy";

echo "DATA_NAME=$DATA_NAME, ATTENTION=$ATTENTION, EXP_NAME=$EXP_NAME";

echo "sbatch --export=EXP_NAME=$EXP_NAME,DATA_NAME=$DATA_NAME,ATTENTION=$ATTENTION --job-name=$EXP_NAME --output=slurm_log/$EXP_NAME.out train.sbatch;"
sbatch --export=EXP_NAME=$EXP_NAME,DATA_NAME=$DATA_NAME,ATTENTION=$ATTENTION --job-name=$EXP_NAME --output=slurm_log/$EXP_NAME.out train.sbatch;
