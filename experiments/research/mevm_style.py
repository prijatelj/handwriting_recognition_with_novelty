"""MEVM for stlye tasks: Writer ID and Repr"""
import argparse
import json
import logging

import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from exputils import io

from hwr_novelty.models.mevm import MEVM
from hwr_novelty.models.style_features import HOG

from experiments.data.iam import HWR



def load_data(datasplit, iam, rimes, hogs):
    #   IAM (known knowns)
    iam_data = HWR(iam.path, datasplit, iam.image_root_dir)

    #   RIMES (known unknowns)
    rimes_data = HWR(rimes.path, datasplit, rimes.image_root_dir)

    # TODO BangalWriting Lines for eval

    # Obtain the labels from the data (writer id)
    images = []
    labels = []
    for item in iam_data:
        images.append(item.image)
        labels.append(item.writer)
    extra_negatives = [item.image for item in rimes_data]
    #bwl_data.df['writer'].values,

    # TODO feature extraction
    #   HOG mean
    #   HOG multi-mean
    #   ResNet 50 (pytorch model repr)
    #   CRNN repr at RNN, at CNN
    #if feature_extraction == 'hog':
    hog = HOG(**hogs.init)
    points = [hog.extract(img, **hogs.extract) for img in images]
    extra_negatives = [
        hog.extract(img, **hogs.extract) for img in extra_negatives
    ]
    #elif feature_extraction in {'resnet50'}:
    #    raise NotImplementedError()
    #else:
    #    raise ValueError(
    #        f'Unexpected value for `feature_extraction`: {feature_extraction}',
    #    )

    return points, labels, extra_negatives


def script_args(parser):
    parser.add_argument(
        'config_path',
        help='YAML experiment configuration file defining the model and data.',
    )

    # TODO eventually will replace with proper config/arg parser
    mevm = parser.add_arg

def parse_args():
    args = io.parse_args(custom_args=script_args)

    # parse yaml config
    with open(args.config_path) as openf:
        config = YAML(typ='safe').load(openf)

    # parse and make data config
    args.data = argparse.Namespace()
    args.data.datasplit = config['data']['datasplit']

    args.data.iam = argparse.Namespace()
    args.data.iam.path = config['data']['iam']['path']
    args.data.iam.image_root_dir = config['data']['iam']['image_root_dir']

    args.data.rimes = argparse.Namespace()
    args.data.rimes.path = config['data']['rimes']['path']
    args.data.rimes.image_root_dir = config['data']['rimes']['image_root_dir']

    # Parse and make HOG config
    args.hogs = argparse.Namespace()

    args.hogs.init = argparse.Namespace()
    args.hogs.init.orientations = config['model']['hogs']['init']['orientations']
    args.hogs.init.pixels_per_cell = config['model']['hogs']['init']['pixels_per_cell']
    args.hogs.init.cells_per_block = config['model']['hogs']['init']['cells_per_block']
    args.hogs.init.block_norm = config['model']['hogs']['init']['block_norm']
    args.hogs.init.feature_vector = config['model']['hogs']['init']['feature_vector']

    args.hogs.extract = argparse.Namespace()
    args.hogs.extract.means = config['model']['hogs']['extract']['means']
    args.hogs.extract.concat_mean = (
        False if 'concat_mean' not in
        config['model']['hogs']['extract']
        else config['model']['hogs']['extract']['means']
    )

    # parse and fill mevm config
    args.mevm = argparse.Namespace()
    args.mevm.save_path = None if 'save_path' not in config['mevm'] else config['mevm']['save_path']
    args.mevm.load_path = None if 'load_path' not in config['mevm'] else config['mevm']['load_path']

    args.mevm.init = argparse.Namespace()
    args.mevm.init.tailsize = config['mevm']['init']['tailsize']
    args.mevm.init.cover_threshold = config['mevm']['init']['cover_threshold']
    args.mevm.init.distance_multiplier = config['mevm']['init']['distance_multiplier']
    args.mevm.init.distance_function = config['mevm']['init']['distance_function']

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
    points, labels, extra_negatives = load_data(
        hogs=args.hogs,
        **vars(args.data),
    )

    # TODO train the MEVMs on augmentated data: elastic transform
    # TODO train the MEVMs on augmentated data: Repr classification task

    #if args.train:
    if args.mevm.save_path:
        logging.info('Training MEVM')
        mevm.fit(points, labels, extra_negatives)
        logging.info('Saving MEVM')
        mevm.save(io.create_filepath(args.mevm.save_path))

    if args.eval and (args.mevm.save_path or args.mevm.load_path):
        #if args.load_probs:
        #    raise NotImplementedError()
        #else:
        # Calc and save the probs
        logging.info('Predicting prob vecs with MEVM')
        probs = mevm.predict(points)

        # Save resulting prob vectors
        logging.info('Saving resulting prob vecs with MEVM')
        df = pd.DataFrame(probs, columns=mevm.labels)
        df.to_csv(**args.output)


        # TODO eval metrics (this should be a generalized and separate script)

        # TODO save argmax then ConfusionMatrix?
        #argmax_probs = probs.argmax(1)
        #conf_mat = ConfusionMatrix(targets, argmax_probs, mevm.labels)

        # TODO calc NMI of on argmax
        # TODO calc k-cluster purity
