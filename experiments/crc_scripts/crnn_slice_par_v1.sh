#!/bin/bash

#$ -pe smp 4
#$ -N crnn_parv2_slice
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/slice/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/crnn/par/slice/logs/

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty/"

# set up the environment
module add conda
conda activate osr_torch

python3 "$BASE_PATH/experiments/research/par_v1/crnn_script.py" \
    "$BASE_PATH/experiments/configs/par_iam_round1/v2/crnn/slice_no_repr.yaml" \
    --slice rnn \
    --log_level DEBUG \
    --eval train test

    # This eval should be train, val, and test, but masks val as test, which is
    # wrong. The point of this is to have a validation set to inform when to
    # stop training the model, or when to save it really.

    #--log_filename "$BASE_PATH/logs/crnn/par/slice/slice.log" \
