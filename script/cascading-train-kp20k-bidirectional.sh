#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=cascading.kp20k.general.copy
#SBATCH --output=slurm_output/cascading.kp20k.general.copy.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore

# Run the job
export ATTENTION="general"
export ROOT_PATH="/zfs1/pbrusilovsky/rum20/seq2seq-keyphrase-pytorch"
export DATA_NAME="kp20k"

export EXP_NAME="cascading.rnn.general.copy"
srun python -m train -data data/$DATA_NAME/$DATA_NAME -vocab_file data/$DATA_NAME/$DATA_NAME.vocab.pt -exp_path "$ROOT_PATH/exp/$EXP_NAME/%s.%s" -exp "$DATA_NAME" -batch_size 16 -bidirectional -run_valid_every 2000 -save_model_every 10000 -beam_size 16 -beam_search_batch_size 8 -train_ml -attention_mode $ATTENTION -copy_attention -copy_mode $ATTENTION -cascading_model
