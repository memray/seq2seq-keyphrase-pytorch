#!/bin/bash
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH -t 00:10:00

srun EXP_NAME = "rnn.scheduled_sampling"
srun echo "Hello $(hostname)"
srun echo "Hello $(EXP_NAME)"