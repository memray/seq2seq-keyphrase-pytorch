#!/usr/bin/env bash

# Run the job
export ATTENTION="general"
export ROOT_PATH="./"
export DATA_NAME="kp20k"
export BATCH_SIZE=1024
export EXP_NAME="rnn.general.copy"
export DEVICE_IDs="0 1 2 3"
export LOG_PATH="./log/$EXP_NAME.log"

nohup python -m train -data data/$DATA_NAME/$DATA_NAME -vocab_file data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "$ROOT_PATH/exp/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size "$BATCH_SIZE" -bidirectional -run_valid_every 10000 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 16 -train_ml -attention_mode $ATTENTION -copy_attention -copy_mode $ATTENTION -device_ids "$DEVICE_IDs" > "$LOG_PATH" &
