#!/bin/bash

#$ -pe smp 16
#$ -N Dcrnnrepr_h5
#$ -q debug
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/crnn_convert/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/crnn_convert/logs/
#$ -t 12-16

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="$BASE_SPLIT/paper_1/writer_id/mevm/hogs"

BASE_EMP="/afs/crc.nd.edu/user/d/dprijate/scratch_21/open_set/hwr_novelty/paper_1/jsons"

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
    echo "Nothing to compute for this one as I have no benchmarks for CRNN."
    exit 0
else
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    OUT_PATH="$BASE_OUT/"split_"$MODO"/mevm_split_"$MODO"_writer_id_no_aug
    MEVM_SAVE="$OUT_PATH"_DUNCE_.hdf5
    #EMB_FP="$BASE_EMP/config$MODO"_"$DATASPLIT"_embeddings.pkl
    EMB_FP="$BASE_EMP/config_rep$MODO"_"$DATASPLIT"_rep_embeddings.pkl
fi

python3 "$BASE_PATH/experiments/research/mevm_style.py" \
    "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
    --log_level INFO \
    --iam_path "$IAM_PATH" \
    --rimes_path "$RIMES_PATH" \
    --output_path "$OUT_PATH"_DUNCE_.csv \
    --output_points "$OUT_PATH"_DUNCE__points.csv \
    --mevm_save "$MEVM_SAVE" \
    --embed_filepath "$EMB_FP"
