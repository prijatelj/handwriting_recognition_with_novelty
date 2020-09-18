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
import character_set
import ctcdecode
import error_rates
import grid_distortion
import hw_dataset
from hw_dataset import HwDataset
import model.crnn as crnn
import string_utils


@dataclass
class DataSet:
    """Contains everything for the dataset handling in one place."""
    idx_to_char: dict
    char_to_idx: dict
    train_dataset: HwDataset
    train_dataloader: DataLoader
    val_dataset: HwDataset
    val_dataloader: DataLoader
    test_dataset: HwDataset
    test_dataloader: DataLoader


def load_data(config, dataset, RIMES=False):
    idx_to_char, char_to_idx = character_set.load_char_set(
        config[dataset]['character_set_path'],
    )

    train_dataset = HwDataset(
        config[dataset]['training_set_path'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config[dataset]['image_root_directory'],
        augmentation=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    try:
        val_dataset = HwDataset(
            config[dataset]['validation_set_path'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config[dataset]['image_root_directory'],
            remove_errors=True,
        )

    except KeyError as e:
        print("No validation set found, generating one")
        master = train_dataset

        print("Total of " +str(len(master)) +" Training Examples")
        n = len(master)  # how many total elements you have
        n_test = int(n * .1)
        n_train = n - n_test

        idx = list(range(n))  # indices to all elements
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        val_dataset = data.Subset(master, test_idx)
        train_dataset = data.Subset(master, train_idx)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    # TODO remove this or figure out why... it even exists?
    """
    if(not RIMES):
        val2_dataset = HwDataset(
            config[dataset]['validation2_set_path'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config[dataset]['image_root_directory'],
            remove_errors=True,
        )

        val2_dataloader = DataLoader(
            val2_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=hw_dataset.collate,
        )
    """

    test_dataset = HwDataset(
        config[dataset]['test_set_path'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config[dataset]['image_root_directory'],
        remove_errors=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=hw_dataset.collate,
    )

    return DataSet(
        idx_to_char,
        char_to_idx,
        train_dataset,
        train_dataloader, # TODO add this, be aware of this
        val_dataset,
        val_dataloader,
        test_dataset,
        test_dataloader,
    )


def eval_crnn_penult(
    hw_crnn,
    dataloader,
    idx_to_char,
    dtype,
    output_crnn_eval=False,
    layer='rnn',
):
    """Evaluates CRNN and returns the rnn_output as the penultimate layer.

    Parameters
    ----------
    hw_crnn :
        The
    dataloader :
        Pytorch dataloader for input to CRNN
    dtype : type
        Pytorch type of a the handwritten line image data. Used to handle CPU
        or GPU use.
    output_crnn_eval : bool, optional
        Outputs the CRNNs performance without the EVM.
    layer : str, optional
        If 'rnn', uses the CRNN's final RNN output as input to the MultipleEVM.
        If 'conv', uses the final convolutional layer's output. If 'concat',
        then returns both concatenated together (Concat is to be implemented).

    Returns
    -------
    list(np.ndarray)
        Returns a list of the selected layer's output for each input sample.
        `layer` determines which layer of the CRNN is used. The shape of each
        np.ndarray is [glyph_window, classes]. This assumes batch size is
        always 1.
    """
    # Initialize metrics
    if output_crnn_eval:
        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.t0

    hw_crnn.eval()

    layer_outs = []

    for x in dataloader:
        if x is None:
            continue
        with torch.no_grad():
            line_imgs = Variable(
                x['line_imgs'].type(dtype),
                requires_grad=False,
            )
            if layer.lower() == 'rnn':
                preds, layer_out = hw_crnn(line_imgs, return_rnn=True)

                # Shape is then [timesteps, hidden layer width]
                layer_outs.append(layer_out.data.cpu().numpy())

            elif layer.lower() == 'conv':
                # Last Convolution Layer
                preds, layer_out = hw_crnn(line_imgs, return_conv=True)

                # Shape is then [timesteps, conv layer flat: height * width]
                layer_outs.append(layer_out.data.cpu().numpy())
                layer_outs.append(np.squeeze(
                    layer_out.permute(1, 0, 2).data.cpu().numpy()
                ))
            else:
                raise NotImplementedError('Concat/both RNN and Conv of CRNN.')

            """
            print(f'line_imgs.shape = {line_imgs.shape}')
            print(f'line_imgs = {line_imgs}')
            print(f'Preds shape: {preds.shape}')
            print(f'RNN Out: {layer_out}')
            print(f'RNN Out: {layer_out.shape}')
            print(f'x["gt"] = {x["gt"]}')
            print(f'x["gt"] len = {len(x["gt"][0])}')
            """
            # Swap 0 and 1 indices to have:
            #   batch sample, "character window", classes
            # Except, since batch sample is always 1 here, that dim is removed:
            #   "character windows", classes

            if output_crnn_eval:
                # TODO save CRNN output for ease of eval and comparison
                output_batch = preds.permute(1, 0, 2)
                out = output_batch.data.cpu().numpy()

                # Consider MEVM input here after enough obtained to do batch
                # training Or save the layer_outs to be used in training the MEVM

                for i, gt_line in enumerate(x['gt']):
                    logits = out[i, ...]

                    pred, raw_pred = string_utils.naive_decode(logits)
                    pred_str = string_utils.label2str(pred, idx_to_char, False)

                    wer = error_rates.wer(gt_line, pred_str)
                    sum_wer += wer

                    cer = error_rates.cer(gt_line, pred_str)

                    tot_we += wer * len(gt_line.split())
                    tot_ce += cer * len(u' '.join(gt_line.split()))

                    sum_loss += cer

                    steps += 1

    if output_crnn_eval:
        message = ''
        message = message + "\nTest CER: " + str(sum_loss / steps)
        message = message + "\nTest WER: " + str(sum_wer / steps)

        print('CRNN results:')
        print("Validation CER", sum_loss / steps)
        print("Validation WER", sum_wer / steps)

        print("Total character Errors:", tot_ce)
        print("Total word errors", tot_we)

        tot_ce = 0.0
        tot_we = 0.0
        sum_loss = 0.0
        sum_wer = 0.0
        steps = 0.0

    return layer_outs


def train_mevm(hw_crnn, mevm, dsets, dtype, positive='iam', cpus=None):
    """Given CRNN model and data, train the mevm on the data."""

    #print('Set Sizes including Nones:')
    #print(f'Validation Set Size = {len(val_dataloader)}')
    #print(f'Test Set Size = {len(test_dataloader)}')

    # Loop through datasets and obtain their layer_out encoding
    layer_outs = {
        k: eval_crnn_penult(hw_crnn, v.train_dataloader, v.idx_to_char, dtype)
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

    with open(config_path) as f:
        config = json.load(f)

    with open(config_path) as f:
        paramList = f.readlines()

    baseMessage = ""

    for line in paramList:
        baseMessage = baseMessage + line

    # Load Data
    datasets = {}
    if 'iam' in config:
        iam_dset = load_data(config, 'iam')
    else:
        raise NotImplementedError('Need to have IAM dataset in config.')

    if 'rimes' in config:
        datasets['rimes'] = load_data(config, 'rimes')
    if 'manuscript' in config:
        datasets['manuscript'] = load_data(config, 'manuscript')

    # Combine the char to idx and idx to chars iam_dset = datasets.pop('iam')
    all_dsets = deepcopy(iam_dset)
    inc = len(all_dsets.char_to_idx) + 1
    for dset in datasets:
        # Remove all char keys that already exist in char to idx
        for key in (
            all_dsets.char_to_idx.keys() & datasets[dset].char_to_idx.keys()
        ):
            datasets[dset].char_to_idx.pop(key, None)

        # Recreate indices based on remaining chars to indices
        for key in datasets[dset].char_to_idx:
            datasets[dset].char_to_idx[key] = inc
            inc += 1

        all_dsets.char_to_idx.update(datasets[dset].char_to_idx)

    datasets['iam'] = iam_dset

    # Create idx_to_char from char_to_idx
    all_dsets.idx_to_char = {v: k for k, v in all_dsets.char_to_idx.items()}

    # NOTE possiblity of these needing to be from iam_dset
    char_to_idx = all_dsets.char_to_idx
    idx_to_char = all_dsets.idx_to_char

    # ??? combine train, val, and test to form knowns and unknowns
    # datasets, dataloaders
    # The datasets could be as simple as a map to the filepaths and just stack
    # the 3 datasets lists together

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

    print(all_dsets.char_to_idx)
    voc = " "
    for x in range(1, len(all_dsets.idx_to_char) + 1):
        voc = voc + all_dsets.idx_to_char[x]
    print(voc)
    print(all_dsets.idx_to_char)

    # NOTE train loop is the CRNN's validation loop, but saving the layer_out
    # results only and then padding as necessary and feeding to the MEVM for
    # training where iam classes is positive (idx 1-79) and rest is negative.

    # Init MEVM from config
    mevm = MEVM(**config['mevm']['init'])

    # Train MEVM given trained CRNN, skip if loading MEVM
    if 'save_path' in config['mevm'] and 'load_path' not in config['mevm']:
        # Train MEVM
        mevm = train_mevm(hw_crnn, mevm, datasets, dtype, cpus=config['cpus'])

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
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)

    # TODO Eval MEVM on val
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)

    # TODO Eval MEVM on test
    #preds = eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets)


if __name__ == "__main__":
    main()
