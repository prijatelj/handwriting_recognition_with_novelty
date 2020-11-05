"""Wraps the MultipleEVM providing functions and functoinality expected from a
SpervisedLearner. Also adds a NominalDataEncoder to map its inputs to its
interal EVMs per class.
"""
import logging

from evm_based_novelty_detector.MultipleEVM import MultipleEVM
from exputils.data.labels import NominalDataEncoder

from hwr_novelty.predictor import SupervisedLearner


class MEVM(MultipleMEVM, SupervisedLearner):
    def __init__(self, *args, **kwargs):
        super(MEVM, self).__init__(*args, **kwargs)

        # Create a NominalDataEncoder to map class inputs to the MEVM internal
        # class represntation.
        self.encoder = NominalDataEncoder([evm.label for evm in self.models])

    def save(self, h5):
        """Performs the same save functionality as in MultipleEVM but adds a
        dataset for the encoder's ordered labels.
        """
        super(MEVM, self).__init__(*args, **kwargs)
        if self._evms is None:
            raise RuntimeError("The model has not been trained yet")
        # open file for writing; create if not existent
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'w')

        # write EVMs
        for i, evm in enumerate(self._evms):
            evm.save(h5.create_group("EVM-%d" % (i+1)))

        # Write labels for the encoder
        h5['labels'] = list(self.encoder.encoder)

    def load(self, h5):
        """Performs the same lod functionality as in MultipleEVM but loads the
        ordered labels from the h5 file for the encoder.
        """
        if isinstance(h5, str):
            h5 = h5py.File(h5, 'r')

        # load evms
        self._evms = []
        i = 1
        while "EVM-%d" % i in h5:
            self._evms.append(EVM(h5["EVM-%d" % (i)], log_level='debug'))
            i += 1

        # Load the ordered label into the NominalDataEncoder
        self.encoder = NominalDataEncoder(h5['labels'][:])

    def predict(self, points):
        """Wraps the MultipleEVM's max_probabilities and uses the encoder to
        keep labels as expected by the user.
        """
        raise NotImplementedError()

    def fit(self, points, labels=None, extra_negatives=None):
        """Wraps the MultipleEVM's train() and uses the encoder to
        """
        # TODO depend on labels in the encoder, ow. update.

        # TODO optionally allow for fit(x, y) functionality AND class_features
        # as MEVM.train() expects

    # TODO def train(): wrap to update encoder
    # TODO def train_update(): wrap to update encoder

    # TODO basically include as much functionality as necessary for generalized
    # trianing of a MEVM here.


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


class MEVMBasedHWR(object):
    raise NotImplementedError()


class MEVMBasedHWRAdapter():
    raise NotImplementedError()
