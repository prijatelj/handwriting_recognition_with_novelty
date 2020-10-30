#!/bin/bash

#$ -pe smp 24
#$ -N mevm_crnn_cc
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/mevm/par/train/crnn_cc/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/mevm/par/train/crnn_cc/logs/

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty/"

# set up the environment
module add conda
conda activate osr_torch

python3 "$BASE_PATH/experiments/research/par_v1/mevm_with_crnn.py" \
    "$BASE_PATH/experiments/configs/par_iam_round1/v2/mevm/crnn_cc.yaml" \
    --layer rnn \
    --mevm_features load_col_chars \
    --col_chars_path "$HOME/scratch_22/open_set/data/handwritten_text_recognition/PAR/round_1/models/v2/mevm/crnn_sam_25pt/crnn_25pt_train_rnn_repr.hdf5" \
    --log_level DEBUG
