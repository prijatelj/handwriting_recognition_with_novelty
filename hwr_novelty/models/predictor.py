"""Abstract class for predictors."""
from abc import ABC, abstractmethod

from exputils.data.labels import NominalDataEncoder

from hwr_novelty.labels import CharEncoder


class Predictor(ABC):
    """Abstract class for predictors."""
    @abstractmethod
    def save(self, filepath):
        """Given filepath, save the current state of the predictor."""
        # TODO consider `save()`
        pass

    @staticmethod
    @abstractmethod
    def load(filepath):
        """Given filepath, load the saved state of the predictor."""
        # TODO consider `load()` and make it a method of the class so
        # `predictor = Predictor.load(path, *args. **kwargs)`
        pass

    @abstractmethod
    def predict(self, features):
        """Given the current state of the predictor, predict the labels"""

        # TODO predict in batches
        pass

    # NOTE optional def eval()


class SupervisedLearner(Predictor):
    """Abstract class for supervised learning predictors."""

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    @abstractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        pass


class SupervisedTranscripter(SupervisedLearner):
    """Abstract class for supervised learning predictors.

    Attributes
    ----------
    char_enc : CharacterEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.char_enc = CharEncoder(*args, **kwargs)

    @asbtractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """
        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        pass


# TODO MEVM(MultipleMEVM, SupervisedLearner):
#   fit(), predict(), NominalDataEncoder
#   save updated to include a dataset for the labels for NominalDataEncoder

"""
class ANNMEVM(SupervisedTranscripter)
    """The combination of an artificial neural network and an MEVM.

    Attributes
    ----------
    ann
    mevm : MEVM
        The MEVM class that wraps the MultipleEVM class.
    """

    def __init__(self):
        pass


class CRNNMEVM(ANNMEVM):
    """The CRNN and MEVM combined to form a transcripter."""

class MEVMBasedHWRDetector():

class MEVMBasedHWRAdapter():
#"""
