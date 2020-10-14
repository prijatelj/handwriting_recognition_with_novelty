#!/bin/bash

#$ -pe smp 4
#$ -N crnn_parv2
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/train/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/train/logs/
#$ -t 1-8

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"
BASE_CONFIG_PATH="$BASE_PATH/experiments/configs/par_iam_round1/v2/crnn"

# set up the environment
module add conda
conda activate osr_torch

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    echo "Adadelta learn rate 1e-2"
    CONFIG="$BASE_CONFIG_PATH/train_no_repr.yaml"

elif [ "$SGE_TASK_ID" -eq "2" ]; then
    echo "Adadelta learn rate 3e-2"
    CONFIG="$BASE_CONFIG_PATH/train_no_repr_lr3e-4.yaml"

elif [ "$SGE_TASK_ID" -eq "3" ]; then
    echo "Adadelta learn rate 1e-2, continue"
    CONFIG="$BASE_CONFIG_PATH/continue/train_no_repr.yaml"

elif [ "$SGE_TASK_ID" -eq "4" ]; then
    echo "Adadelta learn rate 3e-4, continue"
    CONFIG="$BASE_CONFIG_PATH/continue/train_no_repr_lr3e-4.yaml"

elif [ "$SGE_TASK_ID" -eq "5" ]; then
    echo "Adadelta learn rate 1e-2"
    CONFIG="$BASE_CONFIG_PATH/train_no_repr_rmsprop.yaml"

elif [ "$SGE_TASK_ID" -eq "6" ]; then
    echo "Adadelta learn rate 3e-2"
    CONFIG="$BASE_CONFIG_PATH/train_no_repr_rmsprop_lr3e-4.yaml"

elif [ "$SGE_TASK_ID" -eq "7" ]; then
    echo "Adadelta learn rate 1e-2, continue"
    CONFIG="$BASE_CONFIG_PATH/continue/train_no_repr_rmsprop.yaml"

elif [ "$SGE_TASK_ID" -eq "8" ]; then
    echo "Adadelta learn rate 3e-4, continue"
    CONFIG="$BASE_CONFIG_PATH/continue/train_no_repr_rmsprop_lr3e-4.yaml"

else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi


python3 "$BASE_PATH/experiments/research/par_v1/crnn_script.py" \
    "$CONFIG" \
    --train \
    --log_level DEBUG
