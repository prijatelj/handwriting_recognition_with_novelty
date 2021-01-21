#!/bin/bash

#$ -pe smp 8
#$ -N mevmRC
#$ -q gpu
#$ -l gpu=1
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/crnn/repr/train/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/mevm/crnn/repr/train/logs/
#$ -t 13-23

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="/afs/crc.nd.edu/user/d/dprijate/scratch_21/open_set/hwr_novelty"
BASE_OUT="$BASE_OUT/paper_1/repr/mevm/crnn/pca/16"

if [ "$SGE_TASK_ID" -lt "12" ]; then
    DATASPLIT="train"
elif [ "$SGE_TASK_ID" -lt "18" ]; then
    DATASPLIT="val"
else
    DATASPLIT="test"
fi

# set up the environment
module add conda
conda activate osr_torch

# Unique path addons per run
MODO=$(($SGE_TASK_ID % 6))
if [ "$MODO" -eq "5" ]; then
    echo "None of that."
    exit 0
else
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    OUT_PATH="$BASE_OUT/"split_"$MODO"/split_"$MODO"_crnn_pca
    MEVM="$OUT_PATH".hdf5

    EMB_FP="$OUT_PATH/$DATASPLIT/config_rep$MODO"_"$DATASPLIT"_rep_embeddings_PCA_1000_points.hdf5
    OUT_PATH="$OUT_PATH"_eval

    # --pickle_labels load repr labels from pickle, provide path
    REPR_LABELS="/afs/crc.nd.edu/group/cvrl/scratch_28/openSetHWR/models/rep_models/"
    REPR_LABELS="$REPR_LABELS/$MODO"$DATASPLIT"_labels.pkl"
fi

echo "$DATASPLIT"

echo "$MODO"


if [ "$SGE_TASK_ID" -lt "12" ]; then
    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_repr.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH/$DATASPLIT".csv \
        --mevm_save "$MEVM" \
        --datasplit "$DATASPLIT" \
        --embed_filepath "$EMB_FP" \
        --pickle_labels "$REPR_LABELS"
else
    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_repr.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH/$DATASPLIT".csv \
        --mevm_load "$MEVM" \
        --datasplit "$DATASPLIT" \
        --embed_filepath "$EMB_FP" \
        --pickle_labels "$REPR_LABELS"
fi
