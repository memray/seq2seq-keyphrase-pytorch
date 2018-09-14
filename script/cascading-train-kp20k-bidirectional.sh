#!/usr/bin/env bash

# Run the job
export ATTENTION="general"
export ROOT_PATH="./"
export DATA_NAME="kp20k"
export BATCH_SIZE=64
export BEAM_SEARCH_BATCH_SIZE=5
export EXP_NAME="cascading.rnn.general.copy"
export DEVICE_IDs="9 10"
export LOG_PATH="./exp/log/$EXP_NAME.log"

nohup python -m train -data data/$DATA_NAME/$DATA_NAME -vocab_file data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "$ROOT_PATH/exp/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size "$BATCH_SIZE" -bidirectional -run_valid_every 5000 -report_every 10 -save_model_every 5000 -beam_size 32 -beam_search_batch_size "$BEAM_SEARCH_BATCH_SIZE" -train_ml -attention_mode $ATTENTION -copy_attention_mode $ATTENTION -cascading_model -device_ids "$DEVICE_IDs" > "$LOG_PATH" &
