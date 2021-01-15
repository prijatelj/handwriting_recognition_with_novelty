#!/bin/bash

#$ -pe smp 4
#$ -N crnn_pca
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/eval/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/eval/logs/
#$ -t 6-11

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="$BASE_SPLIT/paper_1/writer_id/mevm/crnn_pca/"

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
    echo "No CRNN RNN encoding for the benchmark set."
    exit 0
else
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    OUT_PATH="$BASE_OUT/"split_"$MODO"/mevm_split_"$MODO"_writer_id_no_aug
    MEVM="$OUT_PATH".hdf5
    OUT_PATH="$OUT_PATH"_eval

    # embed_filepath. If a .pkl, then it converts to HDF5. If HDF5, loads to
    # fit PCA and train MEVM
    EMB="path/to/embedding_either_pkl-to-hdf5_or_hdf5-to-pca"
fi

if [ "$SGE_TASK_ID" -lt "12" ]; then
    # Train and save MEVM, points, and results on train data
    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --mevm_save "$MEVM" \
        --datasplit "$DATASPLIT" \
        --output_path "$OUT_PATH"/"$DATASPLIT".csv \
        --output_points "$OUT_PATH"/"$DATASPLIT"_points.csv \
        --embed_filepath "$EMB_FILEPATH"
else
    # Eval by loading MEVM, saving points and results on eval data
    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --mevm_load "$MEVM" \
        --datasplit "$DATASPLIT" \
        --output_path "$OUT_PATH"/"$DATASPLIT".csv \
        --output_points "$OUT_PATH"/"$DATASPLIT"_points.csv \
        --embed_filepath "$EMB_FILEPATH"
fi
