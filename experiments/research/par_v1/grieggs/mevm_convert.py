"""Converts Sam's mevm state and json dict encoder into an MEVM class object
and resaves the state into a single hdf5.
"""
import argparse
import glob
import json
import os

import exputils

from hwr_novelty.models.mevm import MEVM


def script_args(parser):
    parser.add_argument(
        'mevm_state_path',
        help='filepath to 1 mevm state or a directory.',
    )

    parser.add_argument(
        '--json_path',
        default=None,
        help='filepath to 1 JSON encoder file or a dir of json files.',
    )

    parser.add_argument(
        '-o',
        '--output_path',
        default=None,
        help='filepath to save output to..',
    )

    parser.add_argument(
        '--h5ext',
        default='h5',
        help='HDF5 file extention to look for.',
        choices=['h5', 'hdf5'],
    )

    mevm_args(parser)


def mevm_args(parser):
    mevm_args = parser.add_argument_group('mevm', 'MultipleEVM hyperparams')

    mevm_args.add_argument(
        '--tailsize',
        default=1000,
        type=int,
        help='The tailsize of the MultipleEVM.',
        dest='mevm.tailsize',
    )

    mevm_args.add_argument(
        '--cover_threshold',
        default=None,
        type=float,
        help='The cover threshold of the MEVM.',
        dest='mevm.cover_threshold',
    )

    mevm_args.add_argument(
        '--distance_multiplier',
        default=0.5,
        type=float,
        help='The distance multiplier of the MEVM.',
        dest='mevm.distance_multiplier',
    )

    mevm_args.add_argument(
        '--distance_function',
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='The distance functoin to use in the MEVM.',
        dest='mevm.distance_function',
    )

    mevm_args.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'gpu', 'cpu'],
        help='The device to be used by in the MEVM.',
        dest='mevm.device',
    )


def convert(mevm_kwargs, mevm_path, json_path, output_path=None):
    """Loads the MEVM with the given state and a NominalEncoder with the given
    json and combines them into the MEVM wrapper class and saves the state.
    """
    with open(json_path, 'r') as openf:
        enc_dict = json.load(openf)

    labels = list(enc_dict[0].keys())

    mevm = MEVM(labels, **mevm_kwargs)
    mevm.load(mevm_path, labels)

    if output_path is None:
        output_path, ext = os.path.splitext(mevm_path)
        output_path = os.path.join(output_path, '_mevm_obj.hdf5')
    else:
        outpath, ext = os.path.splitext(output_path)
        if ext == '':
            if output_path[-1] == os.path.sep:
                output_path = os.path.join(output_path, 'mevm_obj.hdf5')
            else:
                output_path = f'{output_path}_mevm_obj.hdf5'

    mevm.save(exputils.io.create_filepath(output_path))


def convert_multi(
    mevm_kwargs,
    mevm_path,
    json_path=None,
    output_path=None,
    h5ext='h5',
    common_prefix='_5folds_train-',
):
    if json_path is None:
        json_path = mevm_path

    jsons = glob.glob(os.path.join(json_path, '*.json'))

    for path in glob.iglob(os.path.join(mevm_path, f'*.{h5ext}')):
        # match the strings given same value around the common str
        parts = path.rpartition(common_prefix)
        json_idx = [
            i for i, j in enumerate(jsons)
            if os.path.splitext(parts[-1])[0] in os.path.splitext(j)[0]
        ]

        if len(json_idx) > 1:
            raise ValueError('There are multiple jsons that match!')
        elif len(json_idx) <= 0:
            raise ValueError('There are no jsons that match!')
        json_idx = json_idx[0]

        if os.path.isdir(output_path):
            outpath = os.path.join(
                output_path,
                f'{os.path.splitext(os.path.split(path)[-1])[0]}_mevm_obj.hdf5'
            )
        elif output_path is None:
            outpath = os.path.join(
                'tmp_mevm_states/',
                f'{os.path.splitext(os.path.split(path)[-1])[0]}_mevm_obj.hdf5'
            )
        else:
            outpath = output_path

        convert(mevm_kwargs, path, jsons[json_idx], outpath)


if __name__ == '__main__':
    args = exputils.io.parse_args(
        'Converts Sams mevm state and json dict encoder into an MEVM hdf5.',
        custom_args=script_args,
    )

    if os.path.isfile(args.mevm_state_path):
        if not os.path.isfile(args.json_path):
            raise ValueError(
                'json_path is not a file when mevm state filepath is a file!',
            )
        convert(
            vars(args.mevm),
            args.mevm_state_path,
            args.json_path,
            args.output_path,
        )
    elif os.path.isdir(args.mevm_state_path):
        if args.json_path is not None and os.path.isdir(args.json_path):
            convert_multi(
                vars(args.mevm),
                args.mevm_state_path,
                args.json_path,
                args.output_path,
                h5ext=args.h5ext,
            )
        else:
            convert_multi(
                vars(args.mevm),
                args.mevm_state_path,
                output_path=args.output_path,
                h5ext=args.h5ext,
            )
    else:
        raise ValueError(' '.join([
            'Either handles a single file pair of MEVM state and JSON or a',
            'directory of jsons and mevm states who share similar naming',
            'schemes.',
        ]))
