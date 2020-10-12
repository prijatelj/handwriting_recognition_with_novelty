"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
# Python default packages
import logging
import os

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
from exputils.data.labels import NominalDataEncoder


def eval_crnn_mevm(hw_crnn, mevm, all_dsets, datasets):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data

    Parameters
    ----------

    """
    preds = None

    return preds


def eval_mevm_slices(points, labels, mevm):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data.
    Calculates only the predicted classes.

    Parameters
    ----------

    """
    # TODO adapt the MEVM to actually be a predictor w/ pred() as expected...
    # TODO figure out why this is done here, or why i am told to do this...?

    probs = mevm.max_probabilities(points)


    # It won't be a np.ndarray of [samples, classes + 1 for uknown]. That'd be too useful.

    logging.debug('probs : \n%s', probs)

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
        perf_slices = h5f['perfect_indices'][:].astype(int)

        print('type of perf_slices: ', type(perf_slices))
        print('dtype of perf_slices: ', perf_slices.dtype)
        print('perf_slices: \n', perf_slices)

        # logits SHOULD be [sample, line_character, classes]
        argmax_logits = np.squeeze(h5f['logits'][:, perf_slices]).argmax(axis=1)
        print('argmax_logits shape = ', argmax_logits.shape)

        # Obtain perfect character embeddings only, this is simplest slice
        layers = np.squeeze(h5f['layer'][:, perf_slices])

        print('layers shape = ', layers.shape)

        # TODO need more efficient loading because this nears memory limit!
        # Esp. if the CRNN becomes more accurate w/ more perfect slices!

    logging.info('Number of perfect slices = %d', len(perf_slices))

    # TODO Load the extra negative (known unknowns) data point layer repr

    logging.info(
        'There are %d perfectly predicted transcript lines to train MEVM.',
        len(perf_slices),
    )
    # Perfect slices is no longer needed, as setup is finalized. # TODO unless
    # eval and saving that eval.
    del perf_slices

    logging.debug('Shape of layers: %s', layers.shape)

    # Organize the layers into lists per character class.
    unique_labels, label_counts = np.unique(argmax_logits, return_counts=True)

    logging.debug('Number of Unique Labels: %d', len(unique_labels))
    logging.debug('The unique labels: %s', unique_labels)
    logging.debug('Label counts: %s', label_counts)

    # Be able to obtain the label from the MEVM's indexing of classes
    nominal_encoder = NominalDataEncoder(unique_labels)

    labels_repr = []

    logging.info('Unique Labels contained within layer encoding:')
    for i, label in enumerate(unique_labels):
        logging.info('%d : %d', label, label_counts[i])

        label_indices = np.where(argmax_logits == label)
        labels_repr.append(torch.tensor(layers[label_indices]))

        logging.debug('Label `%s`\'s indices = %s', label, label_indices)
        logging.debug(
            'Torch tensor shape of label %s = %s',
            label,
            labels_repr[i].shape,
        )

    # Init MEVM from config
    mevm = MEVM(**config['model']['mevm']['init'])

    # Train MEVM given CRNN encoded data points
    if ('save_path' in config['model']['mevm']
        and 'load_path' not in config['model']['mevm']
    ):
        # Train MEVM
        mevm.train(labels_repr, labels=np.array(nominal_encoder.encoder))

        # Save trained mevm
        mevm.save(exputils.io.create_filepath(os.path.join(
            config['model']['mevm']['save_path'],
            'mevm_state.hdf5',
        )))

    elif 'save_path' not in config['mevm'] and 'load_path' in config['mevm']:
        # Load MEVM state from file
        mevm.load(config['mevm']['load_path'])
    else:
        raise KeyError(
            'Missing mevm save_path xor load_path, xor both exist in config'
        )

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
