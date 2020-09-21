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


def train_mevm(hw_crnn, mevm, dsets, dtype, positive='iam', cpus=None):
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
        )
        for k, v in dsets.items()
    }

    # Organize the layer_outs and dsets into positives and negatives

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
    #char_to_idx = all_dsets.char_to_idx
    #idx_to_char = all_dsets.idx_to_char

    # Load Model (CRNN)
    if config['model'] == "crnn":
        print("Using CRNN")
        hw_crnn = crnn.create_model({
            'input_height': config['network']['input_height'],
            'cnn_out_size': config['network']['cnn_out_size'],
            'num_of_channels': 3,
            'num_of_outputs': len(iam_dset.idx_to_char) + 1
        })

    hw_crnn.load_state_dict(torch.load(config['model_load_path']))

    if torch.cuda.is_available():
        hw_crnn.cuda()
        dtype = torch.cuda.FloatTensor
        print("Using GPU")
    else:
        dtype = torch.FloatTensor
        print("No GPU detected")

    #print(all_dsets.char_to_idx)
    #voc = " "
    #for x in range(1, len(all_dsets.idx_to_char) + 1):
    #    voc = voc + all_dsets.idx_to_char[x]
    #print(voc)
    #print(all_dsets.idx_to_char)

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
