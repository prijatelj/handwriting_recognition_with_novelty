"""MEVM for stlye tasks: Writer ID and Repr"""
import argparse
import json
import logging
import os
import pickle

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
            augmenters=augmentation.SplitAugmenters.augs,
        )
        logging.info('Split Augmenting RIMES')
        rimes_data = SplitAugmenters(
            iterable=rimes_data,
            augmenters=augmentation.SplitAugmenters.augs,
        )

        # Obtain the labels from the data (writer id)
        logging.info('Getting Labels from IAM.')
        images = []
        labels = []
        extra_negatives = []
        extra_neg_labels = []
        extra_neg_paths = []
        paths = []
        for item in iam_data:
            if item.represent in augmentation.SplitAugmenters.known_unknowns:
                if (
                    item.image is None
                    or not np.isfinite(item.image).all()
                    or np.isnan(item.image).any()
                ):
                    raise ValueError(f'this image be broke: {item.path}')
                extra_negatives.append(item.image)
                extra_neg_labels.append(item.represent)
                extra_neg_paths.append(item.path)
            else:
                if (
                    item.image is None
                    or not np.isfinite(item.image).all()
                    or np.isnan(item.image).any()
                ):
                    raise ValueError(f'this image be broke: {item.path}')
                images.append(item.image)
                labels.append(item.represent)
                paths.append(item.path)

        logging.info('Getting Labels from RIMES.')
        for item in rimes_data:
            if item.represent in augmentation.SplitAugmenters.known_unknowns:
                if (
                    item.image is None
                    or not np.isfinite(item.image).all()
                    or np.isnan(item.image).any()
                ):
                    raise ValueError(f'this image be broke: {item.path}')
                elif 'train2011-1404/10564.png' in item.path:
                    continue
                extra_negatives.append(item.image)
                extra_neg_labels.append(item.represent)
                extra_neg_paths.append(item.path)
            else:
                if (
                    item.image is None
                    or not np.isfinite(item.image).all()
                    or np.isnan(item.image).any()
                ):
                    raise ValueError(f'this image be broke: {item.path}')
                elif 'train2011-1404/10564.png' in item.path:
                    continue
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
            if (
                item.image is None
                or not np.isfinite(item.image).all()
                or np.isnan(item.image).any()
            ):
                raise ValueError(f'this image be broke: {item.path}')

            images.append(item.image)
            labels.append(item.writer)
            paths.append(item.path)

        logging.info('Setting RIMES as extra_negatives.')
        extra_negatives = []
        extra_neg_paths = []

        for item in rimes_data:
            if (
                item.image is None
                or not np.isfinite(item.image).all()
                or np.isnan(item.image).any()
            ):
                raise ValueError(f'this image be broke: {item.path}')
            elif 'train2011-1404/10564.png' in item.path:
                continue
            extra_negatives.append(item.image)
            extra_neg_paths.append(item.path)

        extra_neg_labels = ['rimes'] * len(extra_negatives)

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
        points = np.array([hog.extract(img) for img in images])
        extra_negatives = np.array([
            hog.extract(img) for img in extra_negatives
        ])

    return points, labels, paths + extra_neg_paths, \
        extra_negatives, extra_neg_labels


def load_crnn_data(
    embed_filepath,
    iam,
    rimes,
    augmentation=None,
    max_timestep=656,
    *args,
    **kwargs,
):
    """Loads the preprocessed CRNN embedding of the points."""
    with open(embed_filepath, 'rb') as openf:
        emb = pickle.load(openf)

    points = []
    labels = []
    paths = []

    extra_negatives = []
    extra_neg_labels = []
    extra_neg_paths = []

    if hasattr(augmentation, 'SplitAugmenters'):
        # TODO Exp 3 REPR/appearances augmentation
        return

    # Writer ID data loading:
    # Transform CRNN RNN layer embeddings into points, padding timesetep w/ 0
    # TODO Error if max_timestep - val.shape[0] < 0, or log info & cut to max
    for key, val in emb.items():
        # Extract the label from the keys, create paths, get points
        if 'train2011-' in key or 'eval2011-' in key:
            extra_negatives.append(
                torch.flatten(torch.nn.functional.pad(
                    torch.squeeze(val),
                    (0, 0, 0, max_timestep - val.shape[0]),
                ))
            )
            extra_neg_labels.append('rimes')
            extra_neg_paths.append(os.path.join(
                rimes.image_root_dir,
                key[::-1].replace('-', os.path.sep, 1)[::-1],
            ))
        else:
            points.append(
                torch.flatten(torch.nn.functional.pad(
                    torch.squeeze(val),
                    (0, 0, 0, max_timestep - val.shape[0]),
                ))
            )
            labels.append(key.split('-')[1])
            paths.append(os.path.join(
                iam.image_root_dir,
                f'{key}.png',
            ))
    points = torch.cat(points).detach().cpu().numpy()
    extra_negatives = torch.cat(extra_negatives).detach().cpu().numpy()

    return points, labels, paths + extra_neg_paths, \
        extra_negatives, extra_neg_labels


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
        '--output_points',
        default=None,
        help='output points filepath.',
        dest='output.points',
    )

    parser.add_argument(
        '--augs_per_item',
        default=None,
        type=int,
        help='Number of augmentations per item.',
    )

    parser.add_argument(
        '--torch_extract',
        default=None,
        help='Specify the feature extraction method using a Torch ANN.',
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
        '--embed_filepath',
        default=None,
        help='Filepath ro CRNN embeddings of points.',
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

    parser.add_argument(
        '--hog_means',
        default=None,
        type=int,
        help='Number of sequential means of HOGs to calculate.',
    )

    parser.add_argument(
        '--hog_concat_mean',
        action='store_true',
        help='Concat the entire mean to the multi-mean HOG feature extraction.'
    )

    parser.add_argument(
        '--hog_additive',
        default=None,
        type=float,
        help='The value to add to the end of the hog means.',
    )

    parser.add_argument(
        '--hog_multiplier',
        default=None,
        type=float,
        help='The value to multiply with the hog means, after addition.',
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

            args.data.augmentation.SplitAugmenters.augs = config['data']['augmentation']['SplitAugmenters']['train']

            # TODO Can ignore val and test for now. Necesary for eval.
            # keep the rest as dictionaries as SplitAugmenters expects it.

            if args.data.datasplit in {'val', 'test'}:
                args.data.augmentation.SplitAugmenters.augs.update(config['data']['augmentation']['SplitAugmenters']['val'])

            if args.data.datasplit == 'test':
                # TODO the update will overwrite the Reflect 0 from eval...
                args.data.augmentation.SplitAugmenters.augs.update(config['data']['augmentation']['SplitAugmenters']['test'])


    # Parse and make HOG config
    if args.torch_extract is not None:
        if args.torch_extract == 'resnet50':
            args.hogs = argparse.Namespace()
            args.hogs.init = argparse.Namespace()
            args.hogs.init.network = 'resnet50'
        else:
            raise NotImplementedError(
                f'args.torch_extract == {args.torch_extract}',
            )
    elif 'hogs' in config['model']:
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

        if args.hog_means is None:
            args.hogs.init.means = config['model']['hogs']['init']['means']
        else:
            args.hogs.init.means = args.hog_means

        if args.hog_concat_mean is False:
            if 'concat_mean' in config['model']['hogs']['init']:
                args.hogs.init.concat_mean = config['model']['hogs']['init']['concat_mean']
            else:
                args.hogs.init.concat_mean = False
        else:
            args.hogs.init.concat_mean = args.hog_concat_mean


        if args.hog_additive is None:
            args.hogs.init.additive = (
                None if 'additive' not in
                config['model']['hogs']['init']
                else config['model']['hogs']['init']['additive']
            )
        else:
            args.hogs.init.additive = args.hog_additive

        if args.hog_multiplier is None:
            args.hogs.init.multiplier = (
                None if 'multiplier' not in
                config['model']['hogs']['init']
                else config['model']['hogs']['init']['multiplier']
            )
        else:
            args.hogs.init.multiplier = args.hog_multiplier

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
    if args.embed_filepath is not None:
        # LOAD CRNN pre-processed repr.
        points, labels, paths, extra_negatives, extra_neg_labels = load_crnn_data(
            args.embed_filepath,
            **vars(args.data),
        )
    else:
        points, labels, paths, extra_negatives, extra_neg_labels = load_data(
            feature_extraction=args.hogs,
            **vars(args.data),
        )

    if args.output.points is not None:
        df = pd.DataFrame(np.concatenate((points, extra_negatives)))

        df['gt'] = labels + extra_neg_labels
        df['path'] = paths

        df = df.set_index('path')

        columns = list(df.columns)
        df = df[[columns[-1]] + columns[:-1]]
        df.to_csv(io.create_filepath(args.output.points), index=True)

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

        extra_neg_probs = mevm.predict(extra_negatives)

        if np.isnan(probs).all() and np.isnan(extra_neg_probs).all():
            raise ValueError('`NaN` exists in ALL probs and extra_neg_probs!')

        # Save resulting prob vectors
        logging.info('Saving resulting prob vecs with MEVM')
        df = pd.DataFrame(
            np.concatenate((probs, extra_neg_probs)),
            columns=mevm.labels.tolist() + ['unknown'],
        )

        df['gt'] = labels + extra_neg_labels
        df['path'] = paths

        df = df.set_index('path')

        columns = list(df.columns)
        df = df[[columns[-1]] + columns[:-1]]

        df.to_csv(io.create_filepath(args.output.path), index=True)

        if np.isnan(probs).any() or np.isnan(extra_neg_probs).any():
            # After saving the results to check which cases worked and which
            # did not, raise the error.
            raise ValueError('`NaN` exists in probs or extra_neg_probs!')

        # TODO possibly set the value to 0 if None??? need to figure out why
        # getting blank/empty probs values for benchmark faithful, split 1 and
        # split 3...


        # TODO eval metrics (this should be a generalized and separate script)

        # TODO save argmax then ConfusionMatrix?
        #argmax_probs = probs.argmax(1)
        #conf_mat = ConfusionMatrix(targets, argmax_probs, mevm.labels)

        # TODO calc NMI of on argmax
        # TODO calc k-cluster purity
