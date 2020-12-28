"""Abstract class for predictors."""
from abc import ABC, abstractmethod

from exputils.data.labels import NominalDataEncoder

from hwr_novelty.labels import CharEncoder


class Stateful(ABC):
    """Abstract class for a stateful object who needs save and load methods."""
    @abstractmethod
    def save(self, filepath):
        """Given filepath, save the current state of the object."""
        # TODO consider `save()`
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load(filepath):
        """Given filepath, load the saved state of the object."""
        # TODO consider `load()` and make it a method of the class so
        # `predictor = Predictor.load(path, *args. **kwargs)`
        raise NotImplementedError()


class Predictor(Stateful):
    """Abstract class for predictors."""
    @abstractmethod
    def predict(self, features):
        """Given the current state of the predictor, predict the labels"""

        # TODO predict in batches
        raise NotImplementedError()

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
        raise NotImplementedError()


class SupervisedClassifier(SupervisedLearner):
    """Abstract class for supervised learning classifiers.

    Attributes
    ----------
    label_enc : NominalDataEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.label_enc = NominalDataEncoder(*args, **kwargs)

    @property
    def labels(self):
        return list(self.label_enc.encoder)


class SupervisedTranscripter(SupervisedClassifier):
    """Abstract class for supervised learning predictors.

    Attributes
    ----------
    char_enc : CharacterEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.char_enc = CharEncoder(*args, **kwargs)
        self.label_enc = self.char_enc

    @abstractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()


class SupervisedClassifier(SupervisedLearner):
    """Abstract class for supervised learning classifiers.

    Attributes
    ----------
    label_enc : NominalDataEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.label_enc = NominalDataEncoder(*args, **kwargs)


class SupervisedTranscripter(SupervisedClassifier):
    """Abstract class for supervised learning predictors.

    Attributes
    ----------
    char_enc : CharacterEncoder
    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    def __init__(self, *args, **kwargs):
        self.char_enc = CharEncoder(*args, **kwargs)
        self.label_enc = self.char_enc

    @abstractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """
        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()


# TODO MEVM(MultipleMEVM, SupervisedLearner):
#   fit(), predict(), NominalDataEncoder
#   save updated to include a dataset for the labels for NominalDataEncoder

#class ANNMEVM(SupervisedTranscripter)
#    """The combination of an artificial neural network and an MEVM.
#
#    Attributes
#    ----------
#    ann
#    mevm : MEVM
#        The MEVM class that wraps the MultipleEVM class.
#    """
#
#    def __init__(self):
#        raise NotImplementedError()
#
#
#class CRNNMEVM(ANNMEVM):
#    """The CRNN and MEVM combined to form a transcripter."""
#
#class MEVMBasedHWRDetector():
#
#class MEVMBasedHWRAdapter():
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        the provided data. This uses the existing state of the predictor.
        """
        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # & kwargs as configs
        raise NotImplementedError()


# TODO MEVM(MultipleMEVM, SupervisedLearner):
#   fit(), predict(), NominalDataEncoder
#   save updated to include a dataset for the labels for NominalDataEncoder

#class ANNMEVM(SupervisedTranscripter)
#    """The combination of an artificial neural network and an MEVM.
#
#    Attributes
#    ----------
#    ann
#    mevm : MEVM
#        The MEVM class that wraps the MultipleEVM class.
#    """
#
#    def __init__(self):
#        raise NotImplementedError()
#
#
#class CRNNMEVM(ANNMEVM):
#    """The CRNN and MEVM combined to form a transcripter."""
#
#class MEVMBasedHWRDetector():
#
#class MEVMBasedHWRAdapter():
