"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
# Python default packages
import logging
import os

# 3rd party packages
import h5py
import numpy as np
from ruamel.yaml import YAML
from scipy import stats
from sklearn.decomposition import PCA
import torch
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader

# External packages but within project
from evm_based_novelty_detector.MultipleEVM import MultipleEVM as MEVM
import exputils.io
from exputils.data.labels import NominalDataEncoder
from exputils.data.confusion_matrix import ConfusionMatrix

from experiments.research.par_v1 import crnn_script, crnn_data
from experiments.research.par_v1.grieggs import string_utils


def predict_mevm(
    crnn_repr,
    mevm,
    char_enc,
    crnn_pass=None,
    prob_novel=False,
):
    """Given a CRNN repr of data and MEVM, evaluate the paired models on the
    provided data

    Parameters
    ----------

    Returns
    -------
    list(np.ndarray(int)), list(np.ndarray(float))
        Predictions of the CRNN and MEVM ordered by dataloader order of lines.
        List entries correspond to line images' predicted transcripts, which
        are represented by a numpy array of character encodings. These include
        the repeats and are not decoded yet.

        Also, the probability for every character by the MEVM is also returned
        as a list of numpy arrays of a float per character in the line.
    crnn_pass : list
        list of the classes who when predicted by the CRNN are outputted as is
        without going through the MEVM.
    prob_novel : bool
    """
    # Feed repr thru MEVM, get max_probs, all maintaining line structure
    probs = []
    preds = []

    # NOTE may be faster if the length of each line is calculated, all
    # flattened, run through MEVM and then "reshape" into a list of the
    # different length lines.
    for i, labels_repr in enumerate(crnn_repr):
        # Find the classes predicted by CRNN that do not go through MEVM
        if crnn_pass is not None:
            argmax_logits = logits[i].argmax(1)
            if len(crnn_pass) > 1:
                idx_pass = np.logical_not(np.logical_or(
                    *[argmax_logits == char_enc.encoder[c] for c in crnn_pass]
                ))
            else:
                idx_pass = argmax_logits != char_enc.encoder[crnn_pass[0]]

            max_probs, mevm_idx = mevm.max_probabilities(
                torch.tensor(labels_repr[idx_pass]),
            )

            # put the MEVM probs and preds back into order w/ pass thru samples
            pred_probs = np.ones(len(labels_repr))
            pred_probs[idx_pass] = max_probs
            probs.append(pred_probs)

            argmax_logits[idx_pass] = np.array(mevm_idx)[:, 0]
            preds.append(argmax_logits)
        else:
            max_probs, mevm_idx = mevm.max_probabilities(
                torch.tensor(labels_repr),
            )

            probs.append(np.array(max_probs))
            preds.append(np.array(mevm_idx)[:, 0])

    return preds, probs


def predict_crnn_mevm(
    crnn,
    mevm,
    dataloader,
    char_enc,
    dtype,
    layer='rnn',
    crnn_pass=None,
):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data

    Parameters
    ----------

    Returns
    -------
    list(np.ndarray(int)), list(np.ndarray(float))
        Predictions of the CRNN and MEVM ordered by dataloader order of lines.
        List entries correspond to line images' predicted transcripts, which
        are represented by a numpy array of character encodings. These include
        the repeats and are not decoded yet.

        Also, the probability for every character by the MEVM is also returned
        as a list of numpy arrays of a float per character in the line.
    crnn_pass : list
        list of the classes who when predicted by the CRNN are outputted as is
        without going through the MEVM.
    """
    # Feed data thru crnn, get repr
    # TODO does not have the column characters! only predicting!
    logits, layer_out = crnn_script.eval_crnn(
        crnn,
        dataloader,
        char_enc,
        dtype,
        layer=layer,
    )

    return predict_mevm(layer_out, mevm, char_enc, crnn_pass)


def mevm_decode(
    preds,
    mevm_enc,
    probs=None,
    unknown_threshold=0,
    unknown_idx=None
):
    """Decodes the resulting timestep output from MEVM."""
    # Convert from the MEVM's index enc to char_enc
    if probs is not None and unknown_threshold > 0 and unknown_idx is not None:
        # apply threshold to probs of each character to determine if unknown.
        # If the probability is below threshold, then it is unknown.

        # NOTE that currently if unknown idx is set, that is the default
        # unknown class,  so if it is set to # then all unknowns are assigned
        # to #
        for i, line in enumerate(preds):
            preds[i] = mevm_enc.decode(line)
            preds[i][line < unknown_threshold] = unknown_idx
    else:
        for i, line in enumerate(preds):
            preds[i] = mevm_enc.decode(line)

    return preds


def decode_timestep_output(preds, char_enc, probs=None):
    """Decodes the timestep output of a model like the CRNN and syncs w/ probs
    if given by calculating the mean probability of the character given its
    repeating sequence of occurence.

    Parameters
    ----------
    preds : list(np.ndarray)
        The list of predicted sequence of characters. The characters may be
        represented as their integer encoding or as a probability vector whose
        argmax indicates the chosen character integer encoding.
    char_enc : CharEncoder
        The character encoder that transforms the given character integer
        encoding into its respective character and back.
    probs : list(np.ndarray), optional
        Only provided when the characters in `preds` is represented as an
        integer encoding, probs represents a probability associated with that
        character. As the predicted character sequence is decoded, the mean of
        the sequential characters' probabilities is calculated and saved as
        that charachter's new associated probability.

    Returns
    -------
    list(str) | (list(str), list(np.ndarray))
        Returns the list of strings representing the transcript text from the
        original character prediction sequence (`preds`). If `probs` is given
        or `preds` is given where each character is represented by a
        probability vector, then along with the transcriptions, the probability
        of each character is also returned. This probability per character is
        the mean probability for the chosen character in a sequence.
    """
    # Decode in sync with probs: mean the sequential chars' probs of novelty
    decoded_preds = []
    if probs is not None:
        if len(np.squeeze(preds[0]).shape) > 1:
            raise ValueError(' '.join([
                'When `probs` is given, preds must be a list of 1-dimensional',
                'numpy arrays.',
            ]))
        # probs is given and preds has a int per character prediction
        decoded_probs = []
        scalar_prob = True
    elif len(np.squeeze(preds[0]).shape) > 1:
        # probs is None and preds has a prob vector per character prediction
        decoded_probs = []
        probs = preds
        preds = [prob.argmax(axis=1) for prob in probs]
        scalar_prob = False
    else:
        # probs is None and preds has a int encoding per character prediction
        decoded_probs = None
        scalar_prob = True

    for i, logit in enumerate(preds):
        pred_data = []
        if decoded_probs is not None:
            probs_data = []
            probs_of_sequence = []

        for j in range(len(logit)):
            if (
                logit[j] != 0
                and not ( j > 0 and logit[j] == logit[j - 1] )
            ):
                pred_data.append(logit[j])

                if decoded_probs is not None:
                    # Add the past character's probs if a past char exists
                    if probs_of_sequence:
                        probs_data.append(np.mean(probs_of_sequence, axis=0))

                    # begin new list for the current character
                    probs_of_sequence = [probs[i][j]]
            elif (
                logit[j] != 0
                and decoded_probs is not None
                and not ( j > 0 and logit[j] != logit[j - 1] )
            ):
                # Saves each char prob across a sequence of same characters
                probs_of_sequence.append(probs[i][j])

        decoded_preds.append(string_utils.label2str(
            pred_data,
            char_enc.encoder.inverse,
            False,
            blank_char=char_enc.blank_char,
            blank=char_enc.blank_idx,
        ))

        if decoded_probs is not None:
            if probs_of_sequence:
                # Handle case where theres is only 1 non-zero predicted
                # character and where the last addition to probs_of_sequence
                # needs added.
                probs_data.append(np.mean(probs_of_sequence, axis=0))

            if scalar_prob:
                decoded_probs.append(np.array(probs_data))
            else:
                decoded_probs.append(np.vstack(probs_data))

    if decoded_probs is None:
        return decoded_preds
    return decoded_preds, decoded_probs


def dist_to_uniformity(prob_vecs, axis=1):
    """Returns the distance to uniformity which is the degree of uncertainty
    for the given discrete probability vectors. This assumes that the given
    discrete probability vectors each represent nominal data whose values or
    events are mutually exclusive. This is the standard assumption in
    traditional classification.

    The distance to uniformity is the Euclidean distance of each
    probability vector to the uniform discrete probability vector within that
    probability simplex (i.e. the center of the probability simplex). The
    dimensions of the probability vector determines the corresponding
    probability simplex. The Euclidean distance is within the
    range [0, 1] due to the fartherst a point in the probability simplex can be
    from the center is 1.

    The distance to uniformity is a point-estimate where the probability vector
    is the single point. This does not provide a probability of uncertainty (or
    credibility if Bayesian) of the probability vector itself. It is the degree
    of uncertainty between the nominal values represented by that probability
    vector.

    An alternative to this is to compare the difference in entropy to the
    maxmimum entropy possible given the uniform discrete probability vector.
    This will work for probability vectors with a small number of dimensions,
    but the maximum entropy possible approches infinity as the number of
    dimensions increases. The Normalized euclidean distance avoids that issue,
    but the reliance on the uniform prob vector of the dimension _may_ be more
    restrictive than the entropy calcultion (I am uncertain on this at the
    moment, I need to go through the math).

    Parameters
    ----------
    prob_vecs : np.ndarray
        A matrix of probability vectors whose distance to uniformity is to be
        calculated. This means that the vectors along the given axis will be
        treated as probability vectors, which means the sum of the vector's
        elements must be equal to 1 and each element must be within the range
        [0, 1]. This function does _not_ perform the check if the vectors are
        actual probability vectors.

    Returns
    -------
    np.ndarray
        An array of distances to uniformity which represent the point-estimate
        degree of uncertainty represented as the distance of the probability
        vector to the uniform probability vector of the same dimensions.
    """
    return np.linalg.norm(
        prob_vecs - np.full(prob_vecs.shape[axis], 1 / prob_vecs.shape[axis]),
        ord=2,
        axis=axis,
    )


def eval_crnn_mevm(
    crnn,
    mevm,
    dataloader,
    char_enc,
    mevm_enc,
    dtype,
    layer='rnn',
    decode='naive',
    unknown_threshold=0,
    crnn_pass=None,
):
    """Given a CRNN and MEVM, evaluate the paired models on the provided data

    Parameters
    ----------
    crnn_pass : list
        list of the classes who when predicted by the CRNN are outputted as is
        without going through the MEVM.
    """
    preds, probs = predict_crnn_mevm(
        crnn,
        mevm,
        dataloader,
        char_enc,
        dtype,
        layer=layer,
        crnn_pass=crnn_pass,
    )

    preds = mevm_decode(preds, mevm_enc, probs, unknown_threshold)

    # Obtain ground truth from dataloader
    ground_truth = [x['gt'][0] for x in dataloader]

    # Obtain CER and WER given groundtruth, preds and
    transcript_results = crnn_data.eval_transcription_logits(
        ground_truth,
        preds,
        char_enc,
        decode,
        argmax=False,
    )

    # TODO calculate conf mat from preds (do this elsewhere, as it tends to
    # depend bounding box annotations or known perfect predictions by the CRNN
    #conf_mat = ConfusionMatrix(
    #    char_enc.decode(nominal_enc.decode(ground_truth)), # needs reformated
    #    char_enc.decode(nominal_enc.decode(preds)),
    #    labels=np.array(char_enc.encoder),
    #)

    return transcript_results


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


def col_chars_crnn_fwd(
    crnn,
    dataloader,
    char_enc,
    dtype,
    layer='rnn',
    repeat=4,
    duplicate=True,
):
    """Given CRNN and dataloader, feed forward data and obtain the layer
    representation for every column of characters. Basically make it so the
    MEVM may match the number of characters outputed by the typical CRNN.

    This is only for when assessing the MEVM or the characters per pixel
    columns
    """
    logits, layer_out, col_chars = crnn_script.eval_crnn(
        crnn,
        dataloader,
        char_enc,
        dtype,
        layer=layer,
        return_col_chars=True,
    )

    # Log the number of layer timesteps that match the length of the str
    count_matches = 0

    # Handle the alignment of col_chars to the CRNN layer repr
    for i in range(len(layer_out)):
        if duplicate:
            layer_out[i] = np.repeat(layer_out[i], repeat, axis=0)

            # TODO this is a hotfix and needs replaced eventually. May cause
            # errors
            if layer_out[i].shape[0] > col_chars[i].shape[0]:
                # Duplicates the last charater to pad the layer to the size of
                # the layer_out
                #col_chars[i] = np.append(
                #    col_chars[i],
                #    col_chars[i][
                #        [-1] * (layer_out[i].shape[0] - col_chars[i].shape[0])
                #    ],
                #)
                layer_out[i] = layer_out[i][:
                    col_chars[i].shape[0] - layer_out[i].shape[0]
                ]

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
            else:
                count_matches += 1
        else:
            # Reduce the cols_char by convultion factor (4)
            # TODO pad the characters by ~ to be a multiple of 4
            col_mod = len(col_chars[i]) % 4
            if 0 != col_mod:
                # NOTE cuts, does not pad!
                col_chars[i] = col_chars[i][: -col_mod]

            # Obtain the mode from each group of 4 characters
            col_chars[i] = stats.mode(
                col_chars[i].reshape([-1, 4])
            )[0].flatten()

    logging.info(
        ' '.join([
            'There were a total of `%d` lines of text transcribed. The total',
            'number of layer representation\'s timesteps that matched the',
            'number of characters in the label string after the constant',
            'multiplier: %d',
        ]),
        len(layer_out),
        count_matches,
    )

    assert len(layer_out) == len(col_chars)

    return layer_out, col_chars


def col_chars_crnn(
    crnn,
    dataloader,
    char_enc,
    dtype,
    layer='rnn',
    repeat=4,
    duplicate=True,
):
    """Given bbox directory, CRNN, and character encoder obtains the layer
    representations of the images. Combines the characters of every string into
    a single vector of characters, paired with their CRNN layer representation.
    """
    layer_out, col_chars = col_chars_crnn_fwd(
        crnn,
        dataloader,
        char_enc,
        dtype,
        layer,
        repeat,
        duplicate,
    )

    layer_out_conc = np.concatenate(layer_out)
    col_chars_conc = np.concatenate(col_chars)

    return organize_data_pts_by_logits(col_chars_conc, layer_out_conc)


def load_col_chars(
    char_enc,
    col_chars_path,
    blank_repr_div=4,
    unknown_char_extra_neg=False,
):
    with h5py.File(col_chars_path, 'r') as hf5:
        nominal_enc = NominalDataEncoder([
            char_enc.encoder[key.rpartition('_')[-1]] for key in hf5.keys()
        ])
        labels_repr = [
            torch.tensor(hf5[dat][:]) for dat in hf5.keys()
        ]

        blank_mevm_idx = nominal_enc.encoder[char_enc.encoder['~']]

        if blank_repr_div is not None:
            labels_repr[blank_mevm_idx] = labels_repr[blank_mevm_idx][:int(len(labels_repr[blank_mevm_idx]) / blank_repr_div)]


        # Handle unknown character as extra_negatives
        if (
            char_enc.unknown_idx in nominal_enc.encoder
            and unknown_char_extra_neg
        ):
            unknown_mevm_idx = nominal_enc.encoder[
                char_enc.encoder['#']
            ]

            extra_negatives = labels_repr[unknown_mevm_idx]

            # Given unknown char is treated as rest of unknowns, remove so
            # MEVM do not treat it as a known class.
            nominal_enc.pop(unknown_mevm_idx)
        else:
            extra_negatives = None

    return nominal_enc, labels_repr, extra_negatives


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
        choices=['perfect_slices', 'col_chars', 'load_col_chars'],
    )

    parser.add_argument(
        '--blank_repr_div',
        default=None,
        type=int,
        help='Divides the number of blank char repr samples by this number.',
    )

    parser.add_argument(
        '--unknown_char_extra_neg',
        action='store_true',
        help=' '.join([
            'Treats the unknown character as the known unknown class and thus',
            'all unknowns are expected to be stored here.',
        ])
    )

    parser.add_argument(
        '--unknown_threshold',
        default=0,
        type=int,
        help='Threshold below which predictions are considered unknown.',
    )

    parser.add_argument(
        '--decode',
        default='naive',
        help='How to decode the CRNN\s output.',
        choices=['naive'],
    )

    parser.add_argument(
        '--crnn_pass',
        default=None,
        nargs='+',
        help='Classes that skip the MEVM fwd pass when predicted by the CRNN.',
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

    if args.mevm_features == 'perfect_slices':
        perf_slices, argmax_logits, layers = load_hdf5_slices(
            config['data']['iam']['encoded']['train'],
        )

        logging.info(
            'There are %d perfectly predicted transcript lines to train MEVM.',
            len(perf_slices),
        )
        # Perfect slices is no longer needed, as setup is finalized. # TODO
        # unless eval and saving that eval.
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

        # PCA using Maximum Likelihod Estimation via Minka
        #pca = PCA('mle')
        """
        pca = PCA(100)

        # TODO fit PCA on ALL of the CRNN layer repr in train.
        all_train_loader, all_test_loader, char_enc = crnn_data.load_data(
            config,
        )

        logits, layer_out = crnn_script.eval_crnn(
            crnn,
            all_train_loader,
            char_enc,
            dtype,
            layer=args.layer,
        )

        logging.info('PCA begin fitting.')
        #pca.fit(np.concatenate([c.numpy() for c in train_labels_repr]))
        pca.fit(np.concatenate(layer_out))

        logging.info('PCA components: %d', pca.n_components_)

        train_labels_repr_pca = [torch.tensor(pca.transform(rep.numpy()))
            for rep in train_labels_repr]
        test_labels_repr_pca = [torch.tensor(pca.transform(rep.numpy()))
            for rep in test_labels_repr]
        #"""

        # TODO set the unknown char to the extra_negatives

        extra_negatives = None


        # TODO CLEAN UP: attempt to save memory by deleting objects...
        del train_dataloader
        del test_dataloader

    elif args.mevm_features == 'load_col_chars':
        char_enc = crnn_data.load_config_char_enc(config)
        train_nominal_enc, train_labels_repr, extra_negatives = load_col_chars(
            char_enc,
            args.col_chars_path,
            args.blank_repr_div,
            args.unknown_char_extra_neg,
        )
    else:
        raise ValueError('Unrecognized value for mevm_features.')

    # Init MEVM from config
    mevm = MEVM(device='cpu', **config['model']['mevm']['init'])

    # Train MEVM given CRNN encoded data points
    if (
        'save_path' in config['model']['mevm']
        and 'load_path' not in config['model']['mevm']
    ):
        # Train MEVM
        mevm.train(
            #train_labels_repr_pca,
            train_labels_repr,
            labels=np.array(train_nominal_enc.encoder),
            extra_negatives=extra_negatives,
        )
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


    if args.eval is None:
        return

    # Eval
    if 'train' in args.eval:
        # TODO make sure the data loader is not shuffling! Error ow.
        # TODO warn if the data loader uses augmentation
        results = eval_crnn_mevm(
            crnn,
            mevm,
            train_dataloader,
            char_enc,
            train_nominal_enc,
            dtype,
            layer=args.layer,
            decode=args.decode,
            threshold=args.unknown_threshold,
        )

        logging.info(
            'train eval performance: CER: %f; WER: %f',
            results.char_error_rate,
            results.word_error_rate,
        )

    if 'test' in args.eval:
        # TODO make sure the data loader is not shuffling! Error ow.
        # TODO warn if the data loader uses augmentation
        results = eval_crnn_mevm(
            crnn,
            mevm,
            test_dataloader,
            char_enc,
            train_nominal_enc,
            dtype,
            layer=args.layer,
            decode=args.decode,
            threshold=args.unknown_threshold,
        )

        logging.info(
            'test eval performance: CER: %f; WER: %f',
            results.char_error_rate,
            results.word_error_rate,
        )

if __name__ == "__main__":
    main()
