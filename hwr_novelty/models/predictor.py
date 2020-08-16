"""Abstract class for predictors."""
from abc import ABC

class SupervisedLearner(ABC):
    """Abstract class for abstract learning predictors.

    """

    # TODO __init__ either sets random state or setups up random state to be
    # created when first fit occurs (implies unable to predict in this case).

    @asbtractmethod
    def fit(self, features, labels):
        """Given the current state of the predictor, continue training given
        this new data. This
        """

        # TODO fit in batches
        # TODO fit incrementally
        # TODO parameterize init, fit, and predict st they may be provided args
        # &kwargs as configs
        pass

    @asbtractmethod
    def predict(self, features):
        """Given the current state of the predictor, predict the labels"""

        # TODO predict in batches
        pass

    @asbtractmethod
    def load_state(self):
        """Given filepath, load the saved state of the predictor."""
        # TODO consider `load()`
        pass

    @asbtractmethod
    def save_state(self, filepath):
        """Given filepath, save the current state of the predictor."""
        # TODO consider `save()`
        pass
