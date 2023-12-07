#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=0-10:00
#SBATCH --output=slurm.%A_%a.out
#SBATCH --error=slurm.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucas.wagner@uol.de
#SBATCH --array=1-3

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
   STEP_COUNT=500000
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
   STEP_COUNT=1000000
elif [ ${SLURM_ARRAY_TASK_ID} -eq 3 ]; then
   STEP_COUNT=1500000
fi

python main.py --training_steps ${STEP_COUNT} --env base_prl --save
