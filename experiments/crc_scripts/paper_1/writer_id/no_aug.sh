#!/bin/bash

#$ -pe smp 4
#$ -N mevmWID
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/train/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/train/logs/
#$ -t 1-6

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="BASE_SPLIT/paper_1/writer_id/mevm"

# set up the environment
module add conda
conda activate osr_torch

# Unique path addons per run
if [ "$SGE_TASK_ID" -eq "1" ]; then
    IAM_PATH="$BASE_IAM/iam_split_0_labels.json"
    RIMES_PATH="$BASE_RIMES/rimes_split_0_labels.json"
    OUT_PATH="$BASE_OUT/split_0/mevm_split_0_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/split_0/mevm_split_0_writer_id_no_aug.hdf5"
elif [ "$SGE_TASK_ID" -eq "2" ]; then
    IAM_PATH="$BASE_IAM/iam_split_1_labels.json"
    RIMES_PATH="$BASE_RIMES/rimes_split_1_labels.json"
    OUT_PATH="$BASE_OUT/split_1/mevm_split_1_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/split_1/mevm_split_1_writer_id_no_aug.hdf5"
elif [ "$SGE_TASK_ID" -eq "3" ]; then
    IAM_PATH="$BASE_IAM/iam_split_2_labels.json"
    RIMES_PATH="$BASE_RIMES/rimes_split_2_labels.json"
    OUT_PATH="$BASE_OUT/split_2/mevm_split_2_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/split_2/mevm_split_2_writer_id_no_aug.hdf5"
elif [ "$SGE_TASK_ID" -eq "4" ]; then
    IAM_PATH="$BASE_IAM/iam_split_3_labels.json"
    RIMES_PATH="$BASE_RIMES/rimes_split_3_labels.json"
    OUT_PATH="$BASE_OUT/split_3/mevm_split_3_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/split_3/mevm_split_3_writer_id_no_aug.hdf5"
elif [ "$SGE_TASK_ID" -eq "5" ]; then
    IAM_PATH="$BASE_IAM/iam_split_4_labels.json"
    RIMES_PATH="$BASE_RIMES/rimes_split_4_labels.json"
    OUT_PATH="$BASE_OUT/split_4/mevm_split_4_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/split_4/mevm_split_4_writer_id_no_aug.hdf5"
elif [ "$SGE_TASK_ID" -eq "6" ]; then
    IAM_PATH="$BASE_SPLIT/grieggs_data/IAM_aachen/train.json"
    RIMES_PATH="$BASE_SPLIT/grieggs_data/RIMES_2011_LINES/training.json"
    OUT_PATH="$BASE_OUT/bfaithful/mevm_bfaithful_writer_id_no_aug.csv"
    MEVM_SAVE="$BASE_OUT/bfaithful/mevm_bfaithful_writer_id_no_aug.hdf5"
else
    echo "ERROR: Unexpected SGE_TASK_ID: $SGE_TASK_ID"
    exit 1
fi

python3 "$BASE_PATH/experiments/research/mevm_style.py" \
    "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
    --log_level INFO \
    --iam_path "$IAM_PATH" \
    --rimes_path "$RIMES_PATH" \
    --output_path "$OUT_PATH" \
    --mevm_save "$MEVM_SAVE"
