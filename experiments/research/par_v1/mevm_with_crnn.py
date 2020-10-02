"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
# Python default packages
import logging

# 3rd party packages
import h5py
import numpy as np
from ruamel.yaml import YAML
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

# External packages but within project
from evm_based_novelty_detector.MultipleEVM import MultipleEVM as MEVM
import exputils.io

from experiments.research.par_v1 import crnn_data


def eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data

    Parameters
    ----------

    """
    preds = None

    return preds


def script_args(parser):
    parser.add_argument(
        'config_path',
        help='YAML experiment configuration file defining the model and data.',
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Expect to train model if given.',
    )

    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='The data splits to be evaluated.',
        choices=['train', 'val', 'test'],
    )

    parser.add_argument(
        '--random_seed',
        default=68,
        type=int,
        help=' '.join([
            'Seed used to ensure the eval is deterministic. Give a negative',
            'value if a nondeterministic eval is desired.',
        ]),
    )


def main():
    args = exputils.io.parse_args(custom_args=script_args)

    if args.random_seed:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

        logging.info('Random seed = %d', args.random_seed)

    with open(args.config_path) as openf:
        config = YAML(typ='safe').load(openf)

    # Load the data
    #iam_dset, all_dsets = crnn_data.load_prepare_data(config)

    # Load the class data point layer representations
    with h5py.File(config['data']['iam']['encoded']['train'], 'r') as h5f:
        perf_slices = h5f['indices'][:]
        layers = h5f['layer'][:]
        argmax_logits = h5f['logits'][:].argmax(axis=1)

    # TODO Load the extra negative (known unknowns) data point layer repr

    logging.info(
        'There are %d perfectly predicted characters to train MEVM.',
        len(perf_slices),
    )

    # Organize the layers into lists per character class.
    unique_labels, label_index, label_counts = np.unique(
        argmax_logits,
        return_counts=True,
        return_index=True,
    )

    # Be able to obtain the label from the MEVM's indexing of classes
    label_to_mevm_idx = {}
    label_to_mevm_idx = {}

    labels_repr = []

    logging.info('Unique Labels contained within layer encoding:')
    for i, label in enumerate(unique_labels):
        logging.info('%d : %d', label, label_counts[i])

        label_to_mevm_idx[label] = i
        label_to_mevm_idx[i] = label
        labels_repr.append(layers[label_index[i]])

    # Init MEVM from config
    mevm = MEVM(**config['model']['mevm']['init'])

    # Train MEVM given CRNN encoded data points
    if ('save_path' in config['model']['mevm']
        and 'load_path' not in config['model']['mevm']
    ):
        # Train MEVM
        mevm.train(labels_repr, labels=unique_labels)

        # Save trained mevm
        mevm.save(config['model']['mevm']['save_path'])

    elif 'save_path' not in config['mevm'] and 'load_path' in config['mevm']:
        # Load MEVM state from file
        mevm.load(config['mevm']['load_path'])
    else:
        raise KeyError(' '.join([
            'Missing evm_save_path xor evm_load_path, xor both exist in',
            'config.'
        ]))

    # TODO Eval
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
