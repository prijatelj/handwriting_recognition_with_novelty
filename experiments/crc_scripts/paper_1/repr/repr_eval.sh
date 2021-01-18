#!/bin/bash

#$ -pe smp 8
#$ -N mevmRheval
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/repr/eval/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/repr/eval/logs/
#$ -t 14-14

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="$BASE_SPLIT/paper_1/repr/mevm/hogs"

# set up the environment
module add conda
conda activate osr_torch

if [ "$SGE_TASK_ID" -lt "12" ]; then
    DATASPLIT="train"
elif [ "$SGE_TASK_ID" -lt "18" ]; then
    DATASPLIT="val"
else
    DATASPLIT="test"
fi

# Unique path addons per run
MODO=$(($SGE_TASK_ID % 6))
if [ "$MODO" -eq "5" ]; then
    IAM_PATH="$BASE_SPLIT/grieggs_data/IAM_aachen/train.json"
    RIMES_PATH="$BASE_SPLIT/grieggs_data/RIMES_2011_LINES/training_2011_gt.json"
    OUT_PATH="$BASE_OUT/bfaithful/mevm_bfaithful_repr_aug"
    MEVM="$OUT_PATH".hdf5
    OUT_PATH="$OUT_PATH"_eval

    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_repr.yaml" \
        --log_level DEBUG \
        --iam_path "$IAM_PATH" \
        --iam_image_root_dir "$BASE_SPLIT/grieggs_data/IAM_aachen/" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH"/"$DATASPLIT".csv \
        --datasplit "$DATASPLIT" \
        --output_points "$OUT_PATH"/"$DATASPLIT"_points.csv \
        --mevm_load "$MEVM"
    exit 0
else
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    OUT_PATH="$BASE_OUT/"split_"$MODO"/mevm_split_"$MODO"_repr_aug
    MEVM="$OUT_PATH".hdf5
    OUT_PATH="$OUT_PATH"_eval
fi

python3 "$BASE_PATH/experiments/research/mevm_style.py" \
    "$BASE_PATH/experiments/configs/paper_1/mevm_repr.yaml" \
    --log_level DEBUG \
    --iam_path "$IAM_PATH" \
    --rimes_path "$RIMES_PATH" \
    --output_path "$OUT_PATH"/"$DATASPLIT".csv \
    --datasplit "$DATASPLIT" \
    --output_points "$OUT_PATH"/"$DATASPLIT"_points.csv \
    --mevm_load "$MEVM"
