"""Abstract class for predictors."""
from abc import ABC


class Predictor(ABC):
    """Abstract class for predictors."""
    @asbtractmethod
    def save(self, filepath):
        """Given filepath, save the current state of the predictor."""
        # TODO consider `save()`
        pass

    @asbtractmethod
    def load(self):
        """Given filepath, load the saved state of the predictor."""
        # TODO consider `load()`
        pass


class SupervisedLearner(Predictor):
    """Abstract class for supervised learning predictors."""

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

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

    @asbtractmethod
    def predict(self, features):
        """Given the current state of the predictor, predict the labels"""

        # TODO predict in batches
        pass
