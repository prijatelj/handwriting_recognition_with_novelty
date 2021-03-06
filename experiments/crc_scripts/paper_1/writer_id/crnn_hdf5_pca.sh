#!/bin/bash

#$ -pe smp 16
#$ -N Ccrnn_pca
#$ -q long@@cvrl
#$ -o $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/crnn_pca/logs/
#$ -e $HOME/scratch_365/open_set/hwr/hwr_novelty/logs/paper/crnn_pca/logs/
#$ -t 7-7

# long@@cvrl
# 18-22

# debug
# 12-16

BASE_PATH="$HOME/scratch_365/open_set/hwr/hwr_novelty"

BASE_SPLIT="/afs/crc.nd.edu/user/d/dprijate/scratch_22/open_set/data/handwritten_text_recognition"

BASE_IAM="$BASE_SPLIT/paper_1/data/iam_splits"
BASE_RIMES="$BASE_SPLIT/paper_1/data/rimes_splits"

BASE_OUT="/afs/crc.nd.edu/user/d/dprijate/scratch_21/open_set/hwr_novelty"
BASE_OUT="$BASE_OUT/paper_1/writer_id/mevm/crnn/pca"

BASE_EMP="/afs/crc.nd.edu/group/cvrl/scratch_28/openSetHWR/models"

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

echo "MODO = $MODO ; DATASPLIT = $DATASPLIT"

if [ "$MODO" -eq "5" ]; then
    echo "Nothing to compute for this one as I have no benchmarks for CRNN."
    exit 0
else
    # TODO setup so it can be for Training or Eval, currently only train is in
    # mind.
    IAM_PATH="$BASE_IAM/"iam_split_"$MODO"_labels.json
    RIMES_PATH="$BASE_RIMES/"rimes_split_"$MODO".json
    #OUT_PATH="$BASE_OUT/"split_"$MODO"/mevm_split_"$MODO"_writer_id_no_aug
    OUT_PATH="$BASE_OUT/"split_"$MODO"/split_"$MODO"_crnn_pca
    MEVM="$OUT_PATH".hdf5
    #OUT_PATH="$OUT_PATH"_eval
    EMB_FP="$BASE_EMP/config$MODO"_"$DATASPLIT"_embeddings.hdf5

    PCA_PATH="$BASE_SPLIT/paper_1/writer_id/mevm/crnn/pca/split_$MODO"/split_"$MODO"_crnn_pca

    # TODO transform val and test w/ respective PCA object.
    # TODO eval of CRNN PCA points
fi

if [ "$SGE_TASK_ID" -lt "12" ]; then

    PCA="$PCA_PATH/train/config$MODO"_train_embeddings_PCA_1000_PCA_state.json

    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH"/"$DATASPLIT" \
        --mevm_save "$MEVM" \
        --datasplit "$DATASPLIT" \
        --embed_filepath "$EMB_FP" \
        --pca_comps 1000 \
        --pca_percent .25 \
        --pca_load "$PCA"
else
    PCA="$PCA_PATH/train/config$MODO"_train_embeddings_PCA_1000_PCA_state.json

    python3 "$BASE_PATH/experiments/research/mevm_style.py" \
        "$BASE_PATH/experiments/configs/paper_1/mevm_writer_id_no_aug.yaml" \
        --log_level INFO \
        --iam_path "$IAM_PATH" \
        --rimes_path "$RIMES_PATH" \
        --output_path "$OUT_PATH"/"$DATASPLIT" \
        --mevm_save "$MEVM" \
        --datasplit "$DATASPLIT" \
        --embed_filepath "$EMB_FP" \
        --pca_comps 1000 \
        --pca_percent .25 \
        --pca_load "$PCA"
fi
