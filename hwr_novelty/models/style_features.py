"""The feature extraction of images for HWR style tasks."""
from abc import ABC, abstractmethod
from inspect import getargspec

import numpy as np
from ruamel.yaml import YAML
from skimage.feature import hog


class FeatureExtractor(ABC):
    """Feature extraction abstract class."""

    @abstractmethod
    def extract(self, sample):
        """Every feature extractor will extract features from a sample or
        multiple samples at once.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the feature extractor state to the given filepath."""
        pass

    @staticmethod
    @abstractmethod
    def load(filepath):
        """Load the feature extractor state from the given filepath."""
        pass


class HOGSExtractor(FeatureExtractor):
    """Histogram of Oriented Gradients feature extractor from images.

    Attributes
    ----------
    See `skimage.feature.hog`
    """
    def __init__(self, *args, **kwargs):
        argspec = getargspec(hog)

        # NOTE atm `skimage.feature.hog` is all positional args.
        #if argspec.defaults is not None:
        #    num_required = len(argspec.args) - len(argspec.defaults)
        #else:
        #    num_required = len(argspec.args)
        num_required = 0
        del argspec.args[0]

        on_kwargs = len(args) == 0

        for i, arg in enumerate(argspec.args):
            if i < num_required:
                # Set the arg to required positional value (no default).
                if not on_kwargs and i < len(args):
                    setattr(self, arg, args[i])
                    on_kwargs = i == len(args) - 1
                else:
                    # allow for kwargs to fill in the
                    setattr(self, arg, kwargs[arg])
                    on_kwargs = True
            else:
                # If defaults exist, set them if no positional arg given.
                if not on_kwargs and i < len(args):
                    setattr(self, arg, args[i])
                elif arg in kwargs:
                    setattr(self, arg, kwargs[arg])
                    on_kwargs = True
                else:
                    setattr(self, arg, argspec.defaults[i - num_required])

    def save(self, filepath):
        """Save the HOG feature extractor state to the given filepath."""
        raise NotImplementedError('Not yet.')

    @staticmethod
    def load(filepath):
        """Load the HOG feature extractor state from the given filepath."""

        ext = os.path.splitext(filepath)[-1]

        if ext != 'yaml':
            raise NotImplementedError(
                'Only loading from a YAML file is supported.',
            )

        return

    def extract(
        self,
        img,
        means=1,
        concat_mean=False,
        #orientations=9,
        #pixels_per_cell=(16, 16),
        #cells_per_block=(4, 4),
        #block_norm='L2',
        #visualize=True, FALSE
        #feature_vector=False,
        #**kwargs,
    ):
        """Forward pass of a single image through the Histogram of Oriented
        Gradients for the style tasks.

        Parameters
        ----------
        img : np.ndarray()
            An image used as input to HOGs with 1 channel expected and whose
            elements range from [0, 1].
        means : int, optional
            The number of equal splits of the image where each split results in a
            mean of their corresponding HOGs. This allows for multiple means across
            variable lengthed images, resulting in more sequence information in the
            line of text represented in the image.
        concat_mean : bool, optional
            Takes the mean of all HOGs and concatenates the other means to the end
            of that one. This only occurs if `means` >= 1, otherwise this does
            nothing and this returns the mean of all HOGs.
        pixels_per_cell : int, optional
            pixels per cell to use in the Histogram of Oriented Gradients
            calculation.
        """
        hog_descriptor = hog(img, **vars(self))

        if means <= 1 or concat_mean:
            # Only one mean of all HOGs for the image
            mean_hog = np.mean(hog_descriptor, axis=1).flatten()

            if not concat_mean:
                return mean_hog

        # Histogram of oriented gradients with multiple means
        indices = np.linspace(0, hog_descriptor.shape[1], means)
        hog_steps = [
            np.mean(hog_descriptor[:, indices[i]:idx], axis=1).flatten()
            for i, idx in enumerate(indices[1:])
        ]

        if concat_mean:
            return np.concatenate([mean_hog] + hog_steps)
        return np.concatenate(hog_steps)

# TODO ANN pretrained (e.g. ResNet50 on ImageNet) repr as an encoding

# TODO CRNN layer repr (CNN or RNN)
