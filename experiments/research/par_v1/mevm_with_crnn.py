"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
# Python default packages
from copy import deepcopy
from dataclasses import dataclass
import json
import sys
import os
import time
import sys
sys.path.insert(0, 'CTCDecoder/src')

from __future__ import print_function

# 3rd party packages
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

# External packages but within project
from evm_based_novelty_detector.MultipleEVM import MEVM

# Internal package modules
from hwr_novelty.models import crnn

from experiments.research.par_v1.grieggs import (
    character_set,
    #ctcdecode,
    error_rates,
    grid_distortion,
    hw_dataset,
    string_utils,
)

from experiments.research.par_v1 import crnn_data, crnn_script


def train_mevm(
    hw_crnn,
    mevm,
    dsets,
    dtype,
    positive='iam',
    layer='rnn',
    cpus=None,
):
    """Given CRNN model and data, train the mevm on the data."""

    #print('Set Sizes including Nones:')
    #print(f'Validation Set Size = {len(val_dataloader)}')
    #print(f'Test Set Size = {len(test_dataloader)}')

    # Loop through datasets and obtain their layer_out encoding
    layer_outs = {
        k: crnn_script.eval_crnn(
            hw_crnn,
            v.train_dataloader,
            v.idx_to_char,
            dtype,
            layer='rnn',
            return_logits=False,
        )
        for k, v in dsets.items()
    }

    # Organize the layer_outs and dsets into lists of points per class
    # TODO list of target labels (characters)

    # Order into [# classes, data pt dim] to form multiple classes' positives
    # and negatives

    # Retrieve positive and flatten along character window dim to have an array
    # with shape = [lines * characters, classes]
    positives = np.concatenate(layer_outs.pop(positive))


    # Combine the negatives into 1 list of arrays and then concatenate
    negatives = []
    for k, v in dsets.items():
        negatives += v
    negatives = np.concatenate(negatives)

    # Collapse the 3rd dim into the 2nd for MEVM :: NO 3rd dim now!
    #positives = positives.reshape(-1, np.prod(positives.shape[1:]))
    #negatives = negatives.reshape(-1, np.prod(negatives.shape[1:]))

    print('Begin training MultipleEVM')
    mevm.train(positives, negatives, parallel=cpus) # TODO parallelize
    print('Finished training MEVM')

    # Probably unnecessary return, due to mevm being an object that is updated
    return mevm


def eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data

    Parameters
    ----------

    """
    preds = None

    return preds


def main():
    torch.manual_seed(68)
    torch.backends.cudnn.deterministic = True

    print(torch.LongTensor(10).random_(0, 10))

    config_path = sys.argv[1]
    RIMES = (config_path.lower().find('rimes') != -1)
    print(RIMES)

    with open(config_path) as openf:
        config = json.load(openf)

    iam_dset, all_dsets = crnn_data.load_prepare_data(config)

    # Load Model (CRNN)
    hw_crnn, dtype = crnn_data.init_CRNN(config)

    # NOTE MEVM train loop is the CRNN's validation loop, but saving the
    # layer_out results only and then padding as necessary and feeding to the
    # MEVM for training where iam classes is positive (idx 1-79) and rest is
    # negative.

    # Init MEVM from config
    mevm = MEVM(**config['mevm']['init'])

    # Train MEVM given trained CRNN, skip if loading MEVM
    if 'save_path' in config['mevm'] and 'load_path' not in config['mevm']:
        # Train MEVM
        mevm = train_mevm(hw_crnn, mevm, all_dsets, dtype, cpus=config['cpus'])

        # Save trained mevm
        mevm.save(config['mevm']['save_path'])

    elif 'save_path' not in config['mevm'] and 'load_path' in config['mevm']:
        # Load MEVM state from file
        mevm.load(config['mevm']['load_path'])
    else:
        raise KeyError(' '.join([
            'Missing evm_save_path xor evm_load_path, xor both exist in',
            'config.'
        ]))

    # TODO Eval MEVM on train
    # if some boolean identifier to eval on train
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)

    # TODO Eval MEVM on val
    # if val in datasets
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)

    # TODO Eval MEVM on test
    # if test in datasets
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)


if __name__ == "__main__":
    main()
