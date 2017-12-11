#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titanx
#SBATCH --job-name=tdr_dag
#SBATCH --output=predict_dag.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Load modules
#module restore


# Run the job

for beam_size in 2 4 6 8 16 32 64 128 256
do
#SBATCH --output=predict_dag${beam_size}.out
export EXP_NAME="rnn.teacher_forcing"
export DATA_NAME="kp20k"
srun python -m predict -train_from model/prelim_copy/kp20k-copy.20171203-034153/kp20k-copy.epoch=3.batch=55088.total_batch=185000.model -data data/kp20k/kp20k -vocab data/kp20k/kp20k.vocab.pt -exp_path "exp/%s.%s" -exp "kp20k" -copy_model -bidirectional -batch_size 16 -beam_search_batch_size 1  -batch_workers 4 -must_appear_in_src -beam_size ${beam_size}
done
