#!/bin/bash

#$ -pe smp 4
#$ -N mevmWh
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/train/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/train/logs/
#$ -t 6-11

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="$BASE_SPLIT/paper_1/writer_id/mevm/hogs"

# set up the environment
module add conda
conda activate osr_torch

# Unique path addons per run
MODO=$(($SGE_TASK_ID % 6))
if [ "$MODO" -eq "5" ]; then
    IAM_PATH="$BASE_SPLIT/grieggs_data/IAM_aachen/train.json"
    RIMES_PATH="$BASE_SPLIT/grieggs_data/RIMES_2011_LINES/training_2011_gt.json"
    OUT_PATH="$BASE_OUT/bfaithful/mevm_bfaithful_writer_id_no_aug"
    MEVM_SAVE="$OUT_PATH".hdf5

    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --iam_image_root_dir "$BASE_SPLIT/grieggs_data/IAM_aachen/" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH".csv \
        --output_points "$OUT_PATH"_points.csv \
        --mevm_save "$MEVM_SAVE"
    exit 0
else
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    OUT_PATH="$BASE_OUT/"split_"$MODO"/mevm_split_"$MODO"_writer_id_no_aug
    MEVM_SAVE="$OUT_PATH".hdf5
fi

python3 "$BASE_PATH/experiments/research/mevm_style.py" \
    "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
    --log_level INFO \
    --iam_path "$IAM_PATH" \
    --rimes_path "$RIMES_PATH" \
    --output_path "$OUT_PATH".csv \
    --output_points "$OUT_PATH"_points.csv \
    --mevm_save "$MEVM_SAVE"
