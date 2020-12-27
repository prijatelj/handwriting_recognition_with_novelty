"""Wraps the MultipleEVM providing functions and functoinality expected from a
SpervisedLearner. Also adds a NominalDataEncoder to map its inputs to its
interal EVMs per class.
"""
import logging

import h5py
import numpy as np
import torch

from evm_based_novelty_detector.MultipleEVM import MultipleEVM
from evm_based_novelty_detector.EVM import EVM
from exputils.data.labels import NominalDataEncoder

from hwr_novelty.models.predictor import SupervisedClassifier


class MEVM(MultipleEVM, SupervisedClassifier):
    def __init__(self, labels=None, max_unknown=None, *args, **kwargs):
        super(MEVM, self).__init__(*args, **kwargs)

        # Create a NominalDataEncoder to map class inputs to the MEVM internal
        # class represntation.
        if isinstance(labels, NominalDataEncoder) or labels is None:
            self.label_enc = labels
        elif isinstance(labels, list) or isinstance(labels, np.ndarray):
            self.label_enc = NominalDataEncoder(labels)
        else:
            raise TypeError(' '.join([
                'Expected `labels` of types: None, list, np.ndarray, or',
                'NominalDataEncoder, not of type {type(labels)}'
            ]))

        self.max_unknown = max_unknown

    def save(self, h5):
        """Performs the same save functionality as in MultipleEVM but adds a
        dataset for the encoder's ordered labels.
        """
        if self._evms is None:
            raise RuntimeError("The model has not been trained yet.")

        # Open file for writing; create if not existent
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'w')

        # Write EVMs
        for i, evm in enumerate(self._evms):
            evm.save(h5.create_group("EVM-%d" % (i+1)))

        # Write labels for the encoder
        if self.label_enc is None:
            logging.info('No labels to be saved.')
        else:
            labels = np.array(self.label_enc.encoder)
            h5.attrs['labels_dtype'] = str(labels.dtype)

            if labels.dtype.type is np.str_ or labels.dtype.type is np.string_:
                h5.create_dataset(
                    'labels',
                    data=labels.astype(object),
                    dtype=h5py.special_dtype(vlen=str),
                )
            else:
                h5['labels'] = labels

        # Write training vars
        for attrib in ['tailsize', 'cover_threshold', 'distance_function',
            'distance_multiplier', 'max_unknown',
        ]:
            h5.attrs[attrib] = getattr(self, attrib)

    @staticmethod
    def load(h5, labels=None, labels_dtype=None, train_hyperparams=None):
        """Performs the same lod functionality as in MultipleEVM but loads the
        ordered labels from the h5 file for the label encoder.
        """
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'r')

        # load evms
        _evms = []
        i = 1
        while "EVM-%d" % i in h5:
            _evms.append(EVM(h5["EVM-%d" % (i)], log_level='debug'))
            i += 1

        # Load the ordered label into the NominalDataEncoder
        if 'labels' in h5.keys():
            if labels is not None:
                logging.info(' '.join([
                    '`labels` key exists in the HDF5 MEVM state file, but',
                    'labels was given explicitly to MEVM.load(). Ignoring the',
                    'labels in the HDF5 file.',
                ]))
                label_enc = NominalDataEncoder(labels)
            else:
                if labels_dtype is None:
                    labels_dtype = np.dtype(h5.attrs['labels_dtype'])
                label_enc = NominalDataEncoder(
                    h5['labels'][:].astype(labels_dtype),
                )
        elif labels is not None:
            label_enc = NominalDataEncoder(labels)
        else:
            logging.warning(' '.join([
                'No `labels` dataset available in given hdf5. Relying on the',
                'evm model\'s labels if they exist. Will fail if the MEVM',
                'state does not have any labels in each of its EVM.',
            ]))

            label_enc = NominalDataEncoder(
                [evm.label for evm in _evms],
            )

        # Load training vars if not given
        if train_hyperparams is None:
            # NOTE Able to specify which to load from h5 by passing a list.
            train_hyperparams = [
                'tailsize',
                'cover_threshold',
                'distance_function',
                'distance_multiplier',
                'max_unknown',
            ]

        if isinstance(train_hyperparams, list):
            train_hyperparams = {
                attr: h5.attrs[attr] for attr in train_hyperparams
            }
        elif not isinstance(train_hyperparams, dict):
            raise TypeError(' '.join([
                '`train_hyperparams` expected type: None, list, or dict, but',
                f'recieved {type(train_hyperparams)}',
            ]))

        mevm = MEVM(label_enc, **train_hyperparams)
        mevm._evms = _evms

        return mevm

    #def train(self, *args, **kwargs):
    #    # NOTE this may be necessary if train or train_update are used instead
    #    # of fit to keep the encoder in sync!
    #    super(MEVM, self).train(*args, **kwargs)
    #    self.label_enc = NominalDataEncoder([evm.label for evm in self._evms])

    def fit(self, points, labels=None, extra_negatives=None):
        """Wraps the MultipleEVM's train() and uses the encoder to
        """
        # If points and labels are aligned sequence pair (X, y): adjust form
        if (
            isinstance(points, np.ndarray)
            and (isinstance(labels, list) or isinstance(labels, np.ndarray))
            and len(points) == len(labels)
        ):
            # Adjust sequence pair into list of torch.Tensors and unique labels
            unique = np.unique(labels)
            labels = np.array(labels)
            points = [torch.Tensor(points[labels == u]) for u in unique]
            labels = unique
        elif isinstance(points, list):
            if all([isinstance(pts, np.ndarray) for pts in points]):
                # If list of np.ndarrays, turn into torch.Tensors
                points = [torch.Tensor(pts) for pts in points]
            elif not all([isinstance(pts, torch.Tensor) for pts in points]):
                raise TypeError(' '.join([
                    'expected points to be of types: list(np.ndarray),',
                    'list(torch.tensor), or np.ndarray with labels as an',
                    'aligned list or np.ndarray',
                ]))
        else:
            raise TypeError(' '.join([
                'expected points to be of types: list(np.ndarray),',
                'list(torch.tensor), or np.ndarray with labels as an',
                'aligned list or np.ndarray',
            ]))

        # Set encoder if labels is not None
        if labels is not None:
            if len(points) != len(labels):
                raise ValueError(' '.join([
                    'The given number of labels does not equal the number of',
                    'classes represented by the list of points.',
                    'If giving an aligned sequence pair of points and labels,',
                    'then ensure `points` is of type `np.ndarray`.',
                ]))

            if self.label_enc is not None:
                logging.debug(
                    '`encoder` is not None and is being overwritten!',
                )

            if isinstance(labels, NominalDataEncoder):
                self.label_enc = labels
            elif isinstance(labels, list) or isinstance(labels, np.ndarray):
                self.label_enc = NominalDataEncoder(labels)
            else:
                raise TypeError(' '.join([
                    'Expected `labels` of types: None, list, np.ndarray, or',
                    'NominalDataEncoder, not of type {type(labels)}'
                ]))

        # Ensure extra_negatives is of expected form (no labels for these)
        if (
            (
                isinstance(extra_negatives, np.ndarray)
                and len(extra_negatives.shape) == 2
            )
            or isinstance(extra_negatives, list)
        ):
            extra_negatives = torch.Tensor(extra_negatives)
        elif not isinstance(extra_negatives, torch.Tensor):
            raise TypeError(' '.join([
                'The extra_negatives must be either None, torch.Tensor of',
                'shape 2, or an object broadcastable to such a torch.Tensor.',
            ]))

        # Points is now list(torch.Tensors) and encoder handled.

        # TODO handle adjust of extra negatives as a list of labels to be known
        # unknowns. For now, expects extra_negatives always of correct type.
        self.train(points, labels, extra_negatives)

    def predict(self, points, return_tensor=False):
        """Wraps the MultipleEVM's class_probabilities and uses the encoder to
        keep labels as expected by the user. Also adjusts the class
        probabilities to include the unknown class.

        Returns
        -------
        np.ndarray
        """
        if isinstance(points, np.ndarray):
            points = torch.Tensor(points)
        elif not isinstance(points, torch.Tensor):
            raise TypeError(
                'expected points to be of type: np.ndarray or torch.Tensor',
            )

        probs = self.class_probabilities(points)

        if return_tensor:
            raise NotImplementedError('Not yet.')

        # Find probability of unknown as its own class
        probs = np.array(probs)
        max_probs_known = probs.max(axis=1)
        unknown_probs = (1 - max_probs_known).reshape(-1, )

        # Scale the rest of the known class probs by max prob known
        probs *= max_probs_known.reshape(-1, 1)

        return np.hstack((probs, unknown_probs))


class CRNNMEVM(object):
    """The CRNN and MEVM combined to form a transcripter.

    Attributes
    ----------
    crnn : CRNN
        The Pytorch CRNN model.
    mevm : MEVM
        The MEVM class that wraps the MultipleEVM class.
    """

    def __init__(crnn_kwargs, mevm_kwargs, char_enc):
        self.crnn = CRNN(**crnn_kwargs)
        self.mevm = MEVM(**mevm_kwargs)
        if isinstance(char_enc, CharEncoder):
            self.char_enc = char_enc
        elif isinstance(char_enc, dict):
            self.char_enc = CharEncoder(**char_enc)
        else:
            raise TypeError(f'Unexpected char_enc type: {type(char_enc)}')

    def fit(self,):
        raise NotImplementedError()

    def predict(self, images):
        raise NotImplementedError()

    def save(self, crnn_filepath, mevm_filepath):
        self.crnn.save(crnn_filepath)
        self.mevm.save(mevm_filepath)

    # TODO current implementation does not make sense for a load func when the
    # init handles this.
    #@staticmethod
    #def load(crnn_filepath, mevm_filepath):
    #    self.crnn = CRNN.load(crnn_filepath)
    #    self.MEVM.load(mevm_filepath)
    #    raise NotImplementedError()
