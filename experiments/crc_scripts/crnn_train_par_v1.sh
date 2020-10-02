#!/bin/bash

#$ -pe smp 8
#$ -N crnn_parv1
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/train/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/train/logs/

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"
#DATA_PATH="$HOME/scratch_22/open_set/data/image_net"

# set up the environment
module add conda
conda activate osr_torch

python3 "$BASE_PATH/experiments/research/par_v1/crnn_script.py" \
    "$BASE_PATH/experiments/configs/par_iam_v1/crnn/par_iam_v1_crnn_train.yaml"
