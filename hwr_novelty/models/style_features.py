"""The feature extraction of images for HWR style tasks."""
from abc import abstractmethod
from inspect import getargspec
import logging
import os

import numpy as np
from ruamel.yaml import YAML
from skimage.feature import hog
from torch.nn import Identity
from torchvision import models

from hwr_novelty.models.predictor import Stateful


#class FeatureExtractor(StatefulIterable):
class FeatureExtractor(Stateful):
    """Feature extraction abstract class."""
    @abstractmethod
    def extract(self, image):
        """Every feature extractor will extract features from a sample or
        multiple samples at once.
        """
        raise NotImplementedError()


class HOG(FeatureExtractor):
    """Histogram of Oriented Gradients feature extractor from images.

    Attributes
    ----------
    See `skimage.feature.hog`
    """
    def __init__(
        self,
        means=1,
        concat_mean=False,
        additive=None,
        multiplier=None,
        *args,
        **kwargs,
    ):
        # Save this class' specific attribs first
        self.means = means
        self.concat_mean = concat_mean
        self.additive = additive
        self.multiplier = multiplier

        # Get the args of skimage.feature.hog
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

        #if ext != 'yaml':
        raise NotImplementedError()
        #    'Only loading from a YAML file is supported.',
        #)

        return

    def extract(
        self,
        img,
        means=None,
        concat_mean=None,
        additive=None,
        multiplier=None,
        filler_value=2,
    ):
        """Forward pass of a single image through the Histogram of Oriented
        Gradients for the style tasks.

        Parameters
        ----------
        img : np.ndarray()
            An image used as input to HOGs with 1 channel expected and whose
            elements range from [0, 1].
        means : int, optional
            The number of equal splits of the image where each split results in
            a mean of their corresponding HOGs. This allows for multiple means
            across variable lengthed images, resulting in more sequence
            information in the line of text represented in the image.
        concat_mean : bool, optional
            Takes the mean of all HOGs and concatenates the other means to the
            end of that one. This only occurs if `means` >= 1, otherwise this
            does nothing and this returns the mean of all HOGs.
        pixels_per_cell : int, optional
            pixels per cell to use in the Histogram of Oriented Gradients
            calculation.
        """
        means = means if means is not None else self.means
        concat_mean = concat_mean if concat_mean is not None else self.concat_mean
        additive = additive if additive is not None else self.additive
        multiplier = multiplier if multiplier is not None else self.multiplier


        # Hotfix, TODO the meta args setting needs to be generalized &
        # optionally set the wrapped func into a dict as an attrib of the class
        kwargs = vars(self).copy()
        kwargs.pop('means', None)
        kwargs.pop('concat_mean', None)
        kwargs.pop('additive', None)
        kwargs.pop('multiplier', None)

        hog_descriptor = hog(img, **kwargs)

        if means <= 1 or concat_mean:
            # Only one mean of all HOGs for the image
            mean_hog = np.mean(hog_descriptor, axis=1).ravel()

            if additive is not None:
                logging.debug('added {additive} to the mean hog.')
                # To avoid vectors whose elements are all near zero
                mean_hog += additive

            if multiplier is not None:
                logging.debug('multiplied {multiplier} to the mean hog.')
                # To avoid vectors whose elements are all near zero
                mean_hog *= multiplier

            if not concat_mean:
                return mean_hog

        # Histogram of oriented gradients with multiple means
        indices = np.round(
            np.linspace(0, hog_descriptor.shape[1], means)
        ).astype(int)
        hog_steps = []
        for i, idx in enumerate(indices[1:]):
            next_mean_hog = np.mean(
                hog_descriptor[:, indices[i]:idx],
                axis=1,
            ).ravel()

            # If a NaN occurs, replace it with some number.
            mask = np.logical_not(np.isfinite(next_mean_hog))
            if mask.any():
                next_mean_hog[mask] = filler_value

            if additive is not None:
                # To avoid vectors whose elements are all near zero
                next_mean_hog += additive

            if multiplier is not None:
                # To avoid vectors whose elements are all near zero
                next_mean_hog *= multiplier

            hog_steps.append(next_mean_hog)

        if concat_mean:
            return np.concatenate([mean_hog] + hog_steps)
        return np.concatenate(hog_steps)


# TODO ANN pretrained (e.g. ResNet50 on ImageNet) repr as an encoding
class TorchANNExtractor(FeatureExtractor):
    """Load a pretrained torch network and obtain the desired layer encoding of
    the input as the feature extraction of that input.
    """
    def __init__(self, network, layer='fc', pretrained=True):
        if not isinstance(network, str):
            raise TypeError(
                f'`network` is expected to be a str, not {type(network)}',
            )
        if not hasattr(models, network):
            raise ValueError('`network` is not a valid torchvision model.')

        self.network = getattr(models, network)(pretrained=pretrained)

        if not hasattr(self.network, layer):
            raise NotImplementedError(
                f'`layer` is not an attribute of `network`. `layer` = {layer}'
            )

        setattr(self.network, layer, Identity())

    def extract(self, image):
        return self.network(image).detach().numpy()

    def save(self, filepath):
        """Save the HOG feature extractor state to the given filepath."""
        raise NotImplementedError('Not yet.')

    @staticmethod
    def load(filepath):
        """Load the HOG feature extractor state from the given filepath."""
        raise NotImplementedError('Not yet.')


# TODO CRNN layer repr (CNN or RNN)
