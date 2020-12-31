"""Classes that augment the given image in different ways."""
from abc import abstractmethod
from collections import OrderedDict
import inspect
import sys

import cv2
import numpy as np
from scipy.interpolate import griddata

from hwr_novelty.models.predictor import Stateful


# TODO make torch DataLoader versions of these that can be chained together.
#   Perhaps, make a generic class that allows the user to give the function
#   `augment()`. And make another class that combines torch.DataLoader with
#   these calsses so that they may be used outside of torch if desired.


class StatefulIterable(Stateful):
    def __init__(self, iterable=None):
        """Iterates through the given iterable, applying the augmentation."""
        try:
            iter(iterable)
        except TypeError:
            if iterable is not None:
                raise TypeError(' '.join([
                    '`iterable` must be an iterable object or None,',
                    f'not {type(iterable)}',
                ]))
        self.iterable = iterable

    def __len__(self):
        """Iterates through the given iterable, applying the augmentation."""
        if self.iterable is None:
            raise TypeError('`iterable` is not set! Cannot get length.')
        return len(self.iterable)

    @abstractmethod
    def __getitem__(self, idx):
        """Iterates through the given iterable, applying the augmentation."""
        if self.iterable is None:
            raise TypeError('`iterable` is not set! Cannot get item.')

    def set_iter(self, iterable):
        """Iterates through the given iterable, applying the augmentation."""
        try:
            iter(iterable)
        except TypeError:
            raise TypeError(
                f'`iterable` must be an iterable object, not {type(iterable)}'
            )
        self.iterable = iterable


class Augmenter(StatefulIterable):
    def __getitem__(self, idx):
        super(Augmenter, self).__getitem__(idx)

        item = self.iterable[idx]
        item.image = self.augment(item.image)
        return item

    @abstractmethod
    def augment(self, image):
        raise NotImplementedError()


class StochasticAugmenter(Augmenter):
    """An augmenter that uses a stochastic augmentation method thus needing its
    random state and number of times the augmentation is applied to each item
    in the original iterable.
    """
    def __init__(
        self,
        augs_per_item,
        include_original=False,
        random_state=None,
        iterable=None,
    ):
        super(StochasticAugmenter, self).__init__(iterable)

        self.augs_per_item = augs_per_item
        self.include_original = include_original

        if (
            random_state is None
            or isinstance(random_state, int)
            or isinstance(random_state, np.random.RandomState)
        ):
            self.random_state = np.random.RandomState(random_state)
        else:
            raise TypeError(' '.join([
                '`random_state` expected to be of type None, int, or',
                'np.random.RandomState, but instead recieved argument of',
                f'type: {type(random_state)}',
            ]))

    def __len__(self):
        return super(StochasticAugmenter, self).__len__() * (
            self.augs_per_item + 1 if self.include_original
            else self.augs_per_item
        )

    def __getitem__(self, idx):
        """Iterates through the original iterable of images and extends that
        iterable to be of length: `len(iterable) * augs_per_item` or
        `augs_per_item + 1` if `include_original` is True. If
        `include_original` is True, then the first `len(iterable)` items are
        the original items, unaugmented.
        """
        super(StochasticAugmenter, self).__getitem__(idx)

        if idx >= len(self):
            raise IndexError(f'Index `{idx}` out of range `{len(self)}`.')

        if self.include_original and idx < len(self.iterable):
            return self.iterable[idx]

        item = self.iterable[idx % len(self.iterable)]
        item.image = self.augment(item.image)
        return item


class ElasticTransform(StochasticAugmenter):
    """Performs the elastic transform on the given images via grid distortion.

    Attributes
    ----------
    augs_per_item : int
    include_original : bool
    mesh_interval : int, tuple(int, int)
        Results in a tuple of the width and height intervals `(height_interval,
        width_interval)`. If given an int, then the width and height intervals
        are both set to the given int.
    mesh_std : float, tuple(float, float)
        Results in a tuple of the width and height standard deviations
        `(height_std, width_std)`. If given a float, then the height and width
        standard deviations are both set to the given float.
    interpolation : str
        Method of interpolation, either linear or cubic. Uses cv2.INTER_LINEAR
        or cv2.INTER_CUBIC respectively.
    interpolation_cv2 : cv2.INTER_LINEAR, cv2.INTER_CUBIC
        The opencv interpolation to apply to the images.
    fit_interval_to_image : bool
    draw_grid_lines : bool
    random_state : None, int, np.random.RandomState
    """
    def __init__(
        self,
        mesh_interval=25,
        mesh_std=3.0,
        interpolation='linear',
        fit_interval_to_image=True,
        draw_grid_lines=False,
        *args,
        **kwargs,
    ):
        super(ElasticTransform, self).__init__(*args, **kwargs)

        if (
            isinstance(mesh_interval, tuple)
            and len(mesh_interval) == 2
            and isinstance(mesh_interval[0], int)
            and isinstance(mesh_interval[1], int)
        ):
            self.mesh_interval = mesh_interval
        elif isinstance(mesh_interval, int):
            self.mesh_interval = (mesh_interval, mesh_interval)
        else:
            raise TypeError(' '.join([
                '`mesh_interval` expected type int or tuple(int, int), not',
                f'{type(mesh_interval)}',
            ]))

        if (
            isinstance(mesh_std, tuple)
            and len(mesh_std) == 2
            and isinstance(mesh_std[0], float)
            and isinstance(mesh_std[1], float)
        ):
            self.mesh_std = mesh_std
        elif isinstance(mesh_std, float):
            self.mesh_std = (mesh_std, mesh_std)
        else:
            raise TypeError(' '.join([
                '`mesh_std` expected type float or tuple(float, float), not',
                f'{type(mesh_std)}',
            ]))

        if interpolation == 'linear':
            self.interpolation_cv2 = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            self.interpolation_cv2 = cv2.INTER_CUBIC
        else:
            raise ValueError(' '.join([
                '`interpolation` expected "linear" or "cubic", not',
                f'{interpolation}.',
            ]))
        self.interpolation = interpolation

        self.fit_interval_to_image = fit_interval_to_image
        self.draw_grid_lines = draw_grid_lines

    def augment(self, image):
        height, width = image.shape[:2]

        if self.fit_interval_to_image:
            # Change interval so it fits the image size
            h_ratio = max(1, round(height / float(self.mesh_interval[0])))
            w_ratio = max(1, round(width / float(self.mesh_interval[1])))

            mesh_interval = (height / h_ratio, width / w_ratio)
        else:
            mesh_interval = self.mesh_interval

        # Get control points
        source = np.mgrid[
            0:height + mesh_interval[0]:mesh_interval[0],
            0:width + mesh_interval[1]:mesh_interval[1]
        ]
        source = source.transpose(1, 2, 0).reshape(-1, 2)

        if self.draw_grid_lines:
            if len(image.shape) == 2:
                color = 0
            else:
                color = np.array([0, 0, 255])
            for src in source:
                image[int(src[0]):int(src[0]) + 1, :] = color
                image[:, int(src[1]):int(src[1]) + 1] = color

        # Perturb source control points
        destination = source.copy()
        source_shape = source.shape[:1]
        destination[:, 0] = destination[:, 0] + self.random_state.normal(
            0.0,
            self.mesh_std[0],
            size=source_shape,
        )
        destination[:, 1] = destination[:, 1] + self.random_state.normal(
            0.0,
            self.mesh_std[1],
            size=source_shape,
        )

        # Warp image
        grid_x, grid_y = np.mgrid[0:height, 0:width]
        grid_z = griddata(
            destination,
            source,
            (grid_x, grid_y),
            method=self.interpolation,
        ).astype(np.float32)

        map_x = grid_z[:, :, 1]
        map_y = grid_z[:, :, 0]

        return cv2.remap(
            image,
            map_x,
            map_y,
            self.interpolation_cv2,
            borderValue=(255, 255, 255),
        )

    def save(self, filepath):
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class Noise(StochasticAugmenter):
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0, std=10, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)

        self.mean = mean
        self.std = std

    def augment(self, image):
        noise = np.zeros(image.shape[:2])
        cv2.randn(noise, self.mean, self.std)
        noise = np.stack([noise]*3, axis=-1)
        result = (image / 255) + noise
        # Ensure the range [0, 255]
        result = (np.minimum(np.maximum(result, 0), 1) * 255).astype('uint8')
        return image + result

    def save(self, filepath):
        # TODO possibly make augmenter not stateful and just added it on a per
        # situation basis, given how simple some of these are, but then again
        # perhaps some of these don't need to be augmenters anyways.
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class Blur(StochasticAugmenter):
    """Gaussian blur the image."""
    def __init__(self, ksize, sigmaX, sigmaY=0, *args, **kwargs):
        super(Blur, self).__init__(*args, **kwargs)

        self.ksize = tuple(ksize)
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def augment(self, image):
        return cv2.GaussianBlur(image, self.ksize, self.sigmaX, self.sigmaY)

    def save(self, filepath):
        # TODO possibly make augmenter not stateful and just added it on a per
        # situation basis, given how simple some of these are, but then again
        # perhaps some of these don't need to be augmenters anyways.
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class InvertColor(Augmenter):
    def __init__(self, iterable=None):
        super(InvertColor, self).__init__(iterable)

    def __getitem__(self, idx):
        return self.augment(self.iterable[idx])

    def augment(self, image):
        return 255 - image

    def save(self, filepath):
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class Reflect(Augmenter):
    def __init__(self, axis, iterable=None):
        super(Reflect, self).__init__(iterable)
        self.axis = axis

    def __getitem__(self, idx):
        return self.augment(self.iterable[idx])

    def augment(self, image):
        return np.flip(image, axis=self.axis)

    def save(self, filepath):
        # TODO possibly make augmenter not stateful and just added it on a per
        # situation basis, given how simple some of these are, but then again
        # perhaps some of these don't need to be augmenters anyways.
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class Antique(StochasticAugmenter):
    """Given the image, assumed to be grayscale/binary with white background,
    the text line image is blended with a random selection from a set of
    background paper images. A slice from the random selected background image
    that fits the line image is applied to the line image. This is to simulate
    text on antique papers.
    """
    def __init__(self, *args, **kwargs):
        super(Antique, self).__init__(*args, **kwargs)
        # TODO
        raise NotImplementedError()

    def augment(self):
        # TODO
        raise NotImplementedError()

    def save(self, filepath):
        # TODO possibly make augmenter not stateful and just added it on a per
        # situation basis, given how simple some of these are, but then again
        # perhaps some of these don't need to be augmenters anyways.
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()


class SplitAugmenters(Augmenter):
    """Given multiple augmenters, performs them equally on the iterable's
    items.  This does not expand the given iterable, but rather modifies
    different items within it as balanced as possible given the augmenters.
    """
    def __init__(
        self,
        augmenters,
        include_original=True,
        iterable=None,
    ):
        super(SplitAugmenters, self).__init__(iterable)

        self.include_original = include_original
        augmenters = OrderedDict(augmenters)

        # TODO multiply the size of the iterable. Giving same text and writer
        # to different backgrounds.

        class_members = {name: value for name, value in inspect.getmembers(
            sys.modules[__name__],
            lambda member: (
                inspect.isclass(member)
                and member.__module__ == __name__
                and issubclass(member, Augmenter)
            ),
        )}
        for key, args in augmenters.items():
            if issubclass(type(args), Augmenter):
                continue
            if key not in class_members:
                raise KeyError(
                    f'Given key for an augmenter that does not exist: {key}'
                )

            augmenters[key] = class_members[key](**augmenters[key])

        self.augmenters = list(augmenters.values())

    def __getitem__(self, idx):
        if self.include_original:
            mod_idx = idx % (len(self.augmenters) + 1)

            if mod_idx == len(self.augmenters):
                item = self.iterable[idx]
                item.represent = 'no_aug'
                return item
        else:
            mod_idx = idx % len(self.augmenters)

        augmenter = self.augmenters[mod_idx]

        item = self.iterable[idx]
        item.image = augmenter.augment(item.image)

        name = type(augmenter).__name__
        if name == 'Reflect':
            name = f'{name}_{augmenter.axis}'
        item.represent = name
        return item

    def augment(self, image, aug):
        return aug.augment(image)

    def save(self, filepath):
        # TODO possibly make augmenter not stateful and just added it on a per
        # situation basis, given how simple some of these are, but then again
        # perhaps some of these don't need to be augmenters anyways.
        raise NotImplementedError()

    @staticmethod
    def load(filepath):
        raise NotImplementedError()

# TODO EffectMap / EffectMask: make it so the above effects only apply to parts
# of the image given some distribution of effect. Binary for on/off, or
# gradient of effect where applicable. e.g. partial noise, partial blur.
# Essentially a mask for the effects.
