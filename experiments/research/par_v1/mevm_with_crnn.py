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

from experiments.research.par_v1 import crnn_script, crnn_data

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

    # Train and eval are unused in this script. trianing and eval is inferred
    # from presence of load and save state of mevm
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
        '--col_chars_path',
        default=None,
        help='The root path to all images characters per pixel column.',
    )

    parser.add_argument(
        '--layer',
        default='rnn',
        help='The layer of the ANN to use for feature representaiton.',
        choices=['rnn', 'cnn', 'conv'],
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

    parser.add_argument(
        '--mevm_features',
        default=None,
        help='Slices whose layer repreesntaiton are to obtained.',
        choices=['perfect_slices', 'col_chars'],
    )


def load_hdf5_slices(
    hdf5_path,
    slice_idx='perfect_indices',
    logits_name='logits',
    layers_name='layer',
):
    # Load the class data point layer representations
    with h5py.File(hdf5_path, 'r') as h5f:
        perf_slices = h5f[slice_idx][:].astype(int)

        print('type of perf_slices: ', type(perf_slices))
        print('dtype of perf_slices: ', perf_slices.dtype)
        print('perf_slices: \n', perf_slices)

        # logits SHOULD be [sample, line_character, classes]
        argmax_logits = np.squeeze(h5f[logits_name][:, perf_slices]).argmax(axis=1)
        print('argmax_logits shape = ', argmax_logits.shape)

        # Obtain perfect character embeddings only, this is simplest slice
        layers = np.squeeze(h5f[layers_name][:, perf_slices])

        print('layers shape = ', layers.shape)

        # TODO need more efficient loading because this nears memory limit!
        # Esp. if the CRNN becomes more accurate w/ more perfect slices!

    # TODO Load the extra negative (known unknowns) data point layer repr

    return perf_slices, argmax_logits, layers


def col_chars_crnn(crnn, dataloader, char_enc, dtype, layer='rnn', repeat=4):
    """Given bbox directory, CRNN, and character encoder obtains the layer
    representations of the images.
    """
    logits, layer_out, col_chars = crnn_script.eval_crnn(
        crnn,
        dataloader,
        char_enc,
        dtype,
        layer=layer,
        return_col_chars=True,
    )

    for i in range(len(layer_out)):
        layer_out[i] = np.repeat(layer_out[i], repeat, axis=0)

        # TODO this is a hotfix and needs replaced eventually. May cause errors
        if layer_out[i].shape[0] > col_chars[i].shape[0]:
            # Duplicates the last charater to pad the layer to the size of the
            # layer_out
            col_chars[i] = np.append(
                col_chars[i],
                col_chars[i][
                    [-1] * (layer_out[i].shape[0] - col_chars[i].shape[0]])
                ],
            )

        elif layer_out[i].shape[0] < col_chars[i].shape[0]:
            # Duplicates the last layer to pad the layer to the size of the
            # pixel columns
            layer_out[i] = np.append(
                layer_out[i],
                layer_out[i][
                    [-1] * (col_chars[i].shape[0] - layer_out[i].shape[0])
                ],
                axis=0,
            )

    layer_out_conc = np.concatenate(layer_out)
    col_chars_conc = np.concatenate(col_chars)

    return organize_data_pts_by_logits(col_chars_conc, layer_out_conc)


def organize_data_pts_by_logits(argmax_logits, layers):
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

        label_indices = np.where(argmax_logits == label)[0]
        labels_repr.append(torch.tensor(layers[label_indices]))

        logging.debug('Label `%s`\'s indices = %s', label, label_indices)
        logging.debug(
            'Torch tensor shape of label %s = %s',
            label,
            labels_repr[i].shape,
        )

    return labels_repr, nominal_encoder


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

    if args.mevm_features == 'perfect_slices':
        perf_slices, argmax_logits, layers = load_hdf5_slices(
            config['data']['iam']['encoded']['train'],
        )

        logging.info(
            'There are %d perfectly predicted transcript lines to train MEVM.',
            len(perf_slices),
        )
        # Perfect slices is no longer needed, as setup is finalized. # TODO unless
        # eval and saving that eval.
        del perf_slices

        logging.debug('Shape of layers: %s', layers.shape)

        labels_repr, nominal_encoder = organize_data_pts_by_logits(
            argmax_logits,
            layers,
        )
    elif args.mevm_features == 'col_chars':
        train_dataloader, test_dataloader, char_enc = crnn_data.load_data(
            config,
            args.col_chars_path,
        )

        crnn, dtype = crnn_data.init_CRNN(config)

        #for dataloader in (train_dataloader, test_dataloader):
        train_labels_repr, train_nominal_enc = col_chars_crnn(
            crnn,
            train_dataloader,
            char_enc,
            dtype,
            layer=args.layer,
        )

        test_labels_repr, test_nominal_enc = col_chars_crnn(
            crnn,
            test_dataloader,
            char_enc,
            dtype,
            layer=args.layer,
        )

        # TODO how to handle train/test nominal encoder differences w/ MEVM?
    else:
        raise ValueError('Unrecognized value for mevm_features.')

    # Init MEVM from config
    mevm = MEVM(**config['model']['mevm']['init'])

    # Train MEVM given CRNN encoded data points
    if (
        'save_path' in config['model']['mevm']
        and 'load_path' not in config['model']['mevm']
    ):
        # Train MEVM
        mevm.train(train_labels_repr, labels=np.array(train_nominal_enc.encoder))
        # labels=np.array(nominal_encoder.encoder)

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
