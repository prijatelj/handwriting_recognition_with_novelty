"""MEVM for stlye tasks: Writer ID and Repr"""
import argparse
import json
import logging

import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import torch

from exputils import io

from hwr_novelty.models.augmentation import ElasticTransform, SplitAugmenters
from hwr_novelty.models.mevm import MEVM
from hwr_novelty.models.style_features import HOG, TorchANNExtractor

from experiments.data.iam import HWR


def load_data(
    datasplit,
    iam,
    rimes,
    feature_extraction,
    image_height=64,
    augmentation=None,
):
    logging.info('Loading IAM')
    #   IAM (known knowns)
    iam_data = HWR(iam.path, datasplit, iam.image_root_dir, image_height)

    logging.info('Loading RIMES')
    #   RIMES (known unknowns)
    rimes_data = HWR(rimes.path, datasplit, rimes.image_root_dir, image_height)

    # TODO BangalWriting Lines for eval


    # Augmentation
    if hasattr(augmentation, 'elastic_transform'):
        logging.info('Elastic Transform IAM')
        iam_data = ElasticTransform(
            iterable=iam_data,
            **vars(augmentation.elastic_transform),
        )

        logging.info('Elastic Transform RIMES')
        rimes_data = ElasticTransform(
            iterable=rimes_data,
            **vars(augmentation.elastic_transform),
        )

    if hasattr(augmentation, 'SplitAugmenters'):
        logging.info('Split Augmenting IAM')
        iam_data = SplitAugmenters(
            iterable=iam_data,
            augmenters=augmentation.SplitAugmenters.train,
        )
        logging.info('Split Augmenting RIMES')
        rimes_data = SplitAugmenters(
            iterable=rimes_data,
            augmenters=augmentation.SplitAugmenters.train,
        )

        # Obtain the labels from the data (writer id)
        logging.info('Getting Labels from IAM.')
        images = []
        labels = []
        extra_negatives = []
        paths = []
        for item in iam_data:
            if item.represent in augmentation.SplitAugmenters.known_unknowns:
                extra_negatives.append(item.image)
            else:
                images.append(item.image)
                labels.append(item.represent)
                paths.append(item.path)

        logging.info('Getting Labels from RIMES.')
        for item in rimes_data:
            if item.represent in augmentation.SplitAugmenters.known_unknowns:
                extra_negatives.append(item.image)
            else:
                images.append(item.image)
                labels.append(item.represent)
                paths.append(item.path)
    else:
        # Obtain the labels from the data (writer id)
        logging.info('Getting Labels from IAM.')
        images = []
        labels = []
        paths = []
        for item in iam_data:
            images.append(item.image)
            labels.append(item.writer)
            paths.append(item.path)

        logging.info('Setting RIMES as extra_negatives.')
        extra_negatives = [item.image for item in rimes_data]

    #bwl_data.df['writer'].values,

    logging.info('Performing Feature Extraction')
    # TODO feature extraction
    #   HOG mean
    #   HOG multi-mean
    #   ResNet 50 (pytorch model repr)
    #   CRNN repr at RNN, at CNN

    if hasattr(feature_extraction.init, 'network'):
        logging.info('Performing Feature Extraction: Pretrained Torch ANN')
        ann = TorchANNExtractor(**vars(feature_extraction.init))
        points = np.concatenate([
            ann.extract(torch.Tensor(
                np.expand_dims(image, 0).transpose([0, 3, 1, 2]).copy()
            ).type(torch.FloatTensor))
            for image in images
        ])
        extra_negatives = np.concatenate([
            ann.extract(torch.Tensor(
                np.expand_dims(image, 0).transpose([0, 3, 1, 2]).copy()
            ).type(torch.FloatTensor))
            for image in extra_negatives
        ])
    else:
        logging.info('Performing Feature Extraction: HOG')
        hog = HOG(**vars(feature_extraction.init))
        points = np.array([
            hog.extract(img, **vars(feature_extraction.extract))
            for img in images
        ])
        extra_negatives = np.array([
            hog.extract(img, **vars(feature_extraction.extract))
            for img in extra_negatives
        ])

    return points, labels, extra_negatives, paths


def script_args(parser):
    parser.add_argument(
        'config_path',
        help='YAML experiment configuration file defining the model and data.',
    )

    out = parser.add_argument_group('output', 'Output config')
    parser.add_argument(
        '--output_path',
        default=None,
        help='output filepath.',
        dest='output.path',
    )

    parser.add_argument(
        '--augs_per_item',
        default=None,
        type=int,
        help='Number of augmentations per item.',
    )

    # TODO eventually will replace with proper config/arg parser
    #mevm = parser.add_arg

    # TODO make it easy to replace the iam and rimes split json for CRC script
    data = parser.add_argument_group('data', 'Data config')
    data.add_argument(
        '--iam_path',
        default=None,
        help='Path to IAM labels.',
        dest='data.iam.path',
    )
    data.add_argument(
        '--iam_image_root_dir',
        default=None,
        help='Path to root dir of IAM line images.',
        dest='data.iam.image_root_dir',
    )

    data.add_argument(
        '--rimes_path',
        default=None,
        help='Path to RIMES labels.',
        dest='data.rimes.path',
    )
    data.add_argument(
        '--rimes_image_root_dir',
        default=None,
        help='Path to root dir of RIMES line images.',
        dest='data.rimes.image_root_dir',
    )

    data.add_argument(
        '--datasplit',
        default=None,
        help='Datasplit to use.',
        dest='data.datasplit',
    )

    parser.add_argument(
        '--mevm_save',
        default=None,
        help='Path to save trained MEVM.',
    )

    parser.add_argument(
        '--mevm_load',
        default=None,
        help='Path to save trained MEVM.',
    )

    parser.add_argument(
        '--tailsize',
        default=None,
        type=int,
        help='Tailsize of MEVM.',
    )

def parse_args():
    args = io.parse_args(custom_args=script_args)

    # parse yaml config
    with open(args.config_path) as openf:
        config = YAML(typ='safe').load(openf)

    # parse and make data config
    #args.data = argparse.Namespace()

    if args.data.datasplit is None:
        if 'datasplit' in config['data']:
            args.data.datasplit = config['data']['datasplit']
        else:
            raise ValueError(
                '`datasplit` must be provided as an arg or in the config',
            )
    args.data.image_height = config['data']['image_height']

    #args.data.iam = argparse.Namespace()
    if args.data.iam.path is None:
        args.data.iam.path = config['data']['iam']['path']
    if args.data.iam.image_root_dir is None:
        args.data.iam.image_root_dir = config['data']['iam']['image_root_dir']

    #args.data.rimes = argparse.Namespace()
    if args.data.rimes.path is None:
        args.data.rimes.path = config['data']['rimes']['path']
    if args.data.rimes.image_root_dir is None:
        args.data.rimes.image_root_dir = config['data']['rimes']['image_root_dir']

    # Augmentation
    if 'augmentation' in config['data']:
        args.data.augmentation = argparse.Namespace()
        # TODO want to perform these in order, so a list or ordered
        # dict/namespace would be good.

        if 'elastic_transform' in config['data']['augmentation']:
            args.data.augmentation.elastic_transform = argparse.Namespace()
            #args.data.augmentation.elastic_transform.mesh_interval = config['data']['augmentation']['elastic_transform']['mesh_interval']
            mesh_std = \
                config['data']['augmentation']['elastic_transform']['mesh_std']
            args.data.augmentation.elastic_transform.mesh_std = (
                float(mesh_std[0]),
                float(mesh_std[1]),
            )

            if args.augs_per_item is not None:
                args.data.augmentation.elastic_transform.augs_per_item = args.augs_per_item
            else:
                args.data.augmentation.elastic_transform.augs_per_item = config['data']['augmentation']['elastic_transform']['augs_per_item']
            args.data.augmentation.elastic_transform.include_original = config['data']['augmentation']['elastic_transform']['include_original']
            args.data.augmentation.elastic_transform.random_state = 0

        if 'SplitAugmenters' in config['data']['augmentation']:
            args.data.augmentation.SplitAugmenters = argparse.Namespace()

            args.data.augmentation.SplitAugmenters.known_unknowns = config['data']['augmentation']['SplitAugmenters']['known_unknowns']

            args.data.augmentation.SplitAugmenters.train = config['data']['augmentation']['SplitAugmenters']['train']

            # TODO Can ignore val and test for now. Necesary for eval.
            # keep the rest as dictionaries as SplitAugmenters expects it.


    # Parse and make HOG config
    if 'hogs' in config['model']:
        args.hogs = argparse.Namespace()

        args.hogs.init = argparse.Namespace()
        args.hogs.init.orientations = config['model']['hogs']['init']['orientations']
        args.hogs.init.pixels_per_cell = config['model']['hogs']['init']['pixels_per_cell']
        args.hogs.init.cells_per_block = config['model']['hogs']['init']['cells_per_block']
        args.hogs.init.block_norm = config['model']['hogs']['init']['block_norm']
        args.hogs.init.feature_vector = config['model']['hogs']['init']['feature_vector']
        args.hogs.init.multichannel = config['model']['hogs']['init']['multichannel']

        ''' Generalized config parsing for models, but missing defaults.
        for model_id, margs in config['model'].items():
            setattr(args, model_id, argparse.Namespace())

            for func, fargs in margs.items():
                model_obj = getattr(args, model_id)
                if func in ['save_path', 'load_path']:
                    setattr(model_obj, func, fargs)
                    continue

                setattr(model_obj, func, argparse.Namespace())

                for attr, value in fargs.items():
                    setattr(getattr(model_obj, func), attr, value)
        #'''

        args.hogs.extract = argparse.Namespace()
        args.hogs.extract.means = config['model']['hogs']['extract']['means']

        args.hogs.extract.concat_mean = (
            False if 'concat_mean' not in
            config['model']['hogs']['extract']
            else config['model']['hogs']['extract']['means']
        )
    elif 'feature_extraction' in config['model']:
        # keeping it hogs cuz hot patch.
        args.hogs = argparse.Namespace()
        args.hogs.init = argparse.Namespace()
        args.hogs.init.network = config['model']['feature_extraction']['init']['network']
    else:
        raise KeyError(
            'missing feature extraction or `hogs` in config `models`.'
        )

    # parse and fill mevm config
    args.mevm = argparse.Namespace()

    if args.mevm_save is None:
        if 'save_path' in config['model']['mevm']:
            args.mevm.save_path = config['model']['mevm']['save_path']
        else:
            args.mevm.save_path = None
    else:
        args.mevm.save_path = args.mevm_save

    if args.mevm_load is None:
        if 'load_path' in config['model']['mevm']:
            args.mevm.load_path = config['model']['mevm']['load_path']
        else:
            args.mevm.load_path = None
    else:
        args.mevm.load_path = args.mevm_load

    args.mevm.init = argparse.Namespace()
    if args.tailsize is None:
        args.mevm.init.tailsize = config['model']['mevm']['init']['tailsize']
    else:
        args.mevm.init.tailsize = args.tailsize
    args.mevm.init.cover_threshold = config['model']['mevm']['init']['cover_threshold']
    args.mevm.init.distance_multiplier = config['model']['mevm']['init']['distance_multiplier']
    args.mevm.init.distance_function = config['model']['mevm']['init']['distance_function']

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.mevm.load_path:
        logging.info('Loading MEVM')
        mevm = MEVM.load(args.mevm.load_path)
    else:
        logging.info('Initalizing MEVM')
        mevm = MEVM(**vars(args.mevm.init))

    # Load data and feature extract
    logging.info('Loading Data')
    points, labels, extra_negatives, paths = load_data(
        feature_extraction=args.hogs,
        **vars(args.data),
    )

    #if args.train:
    if args.mevm.save_path:
        logging.info('Training MEVM')
        mevm.fit(points, labels, extra_negatives)
        logging.info('Saving MEVM')
        mevm.save(io.create_filepath(args.mevm.save_path))

    if args.output.path and (args.mevm.save_path or args.mevm.load_path):
        #if args.load_probs:
        #    raise NotImplementedError()
        #else:
        # Calc and save the probs
        logging.info('Predicting prob vecs with MEVM')
        probs = mevm.predict(points)

        # Save resulting prob vectors
        logging.info('Saving resulting prob vecs with MEVM')
        df = pd.DataFrame(probs, columns=mevm.labels.tolist() + ['unknown'])

        df['gt'] = labels
        df['path'] = paths

        df = df.set_index('path')

        columns = list(df.columns)
        df = df[[columns[-1]] + columns[:-1]]

        df.to_csv(io.create_filepath(args.output.path), index=True)

        # TODO possibly set the value to 0 if None??? need to figure out why
        # getting blank/empty probs values for benchmark faithful, split 1 and
        # split 3...


        # TODO eval metrics (this should be a generalized and separate script)

        # TODO save argmax then ConfusionMatrix?
        #argmax_probs = probs.argmax(1)
        #conf_mat = ConfusionMatrix(targets, argmax_probs, mevm.labels)

        # TODO calc NMI of on argmax
        # TODO calc k-cluster purity
