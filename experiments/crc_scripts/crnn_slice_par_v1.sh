#!/bin/bash

#$ -pe smp 4
#$ -N crnn_parv1_slice
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/slice/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/slice/logs/

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty/"
#DATA_PATH="$HOME/scratch_22/open_set/data/image_net"

# set up the environment
module add conda
conda activate osr_torch

python3 "$BASE_PATH/experiments/research/par_v1/crnn_script.py" \
    "$BASE_PATH/experiments/configs/par_iam_v1/crnn/par_iam_v1_crnn_slice.yaml" \
    --slice rnn \
    --log_level DEBUG \
    --eval train test
