"""Calculate the measures on the data and save the confusion matrices."""
from collections import namedtuple
from glob import glob
import json
import os

import numpy as np
import pandas as pd

from exputils.io import parse_args, create_filepath, NumpyJSONEncoder
from exputils.data import ConfusionMatrix

def script_args(parser):
    # Loading
    # experiment dir
    # model dir
    # train dir # ignore *_points.csv
    # val and test dir # ignore *_points.csv

    parser.add_argument('experiment_dir', help='Directory of experiments.')
    parser.add_argument('models', help='Dirs of models in experiment dir.')
    parser.add_argument(
        '--unknowns',
        default=None,
        help='labels treated as unknown.',
    )

    parser.add_argument(
        '--no_labels',
        action='store_true',
        help='No labels in the probs dataframes.'
    )


def get_dfs(experiment_dir, models):
    paths = []
    dfs = []
    Probs = namedtuple('Probs', ['model', 'train', 'val', 'test'])
    DFP = namedtuple('DFPath', ['split', 'path', 'df'])

    for model in models:
        path = os.path.join(experiment_dir, model)
        if os.path.isdir(path):
            train = glob(f'{path}/*/*[!_points].csv')
            val = glob(f'{path}/*/*/val.csv')
            test = glob(f'{path}/*/*/test.csv')

            dfs.append(Probs(
                model,
                DFP('train', train, [pd.read_csv(path) for path in train]),
                DFP('val', val, [pd.read_csv(path) for path in val]),
                DFP('test', test, [pd.read_csv(path) for path in test]),
            ))
        else:
            raise IOError(f'Filepath does not exist: {path}')

    return dfs


if __name__ == '__main__':
    args = parse_args(custom_args=script_args)

    args.models = args.models.split(' ')
    if args.unknowns is not None:
        args.unknowns = args.unknowns.split(' ')

    # Load the probs csvs
    prob_dfs = get_dfs(args.experiment_dir, args.models)

    # Create a confusion matrix for each probs csv and save
    results = {}
    for dfs in prob_dfs:
        tmp_res = []
        tmp_nd = []
        splits = {}

        # Skip first since that is the model str.
        for split, path, df in dfs[1:]:
            pred = df[df.columns[2:]].values.argmax(1)
            pred = list(df.columns[pred])

            missed_labels = list(set(df['gt']) - set(df.columns[2:]))
            labels = list(df.columns[2:]) + missed_labels

            cm = ConfusionMatrix(df['gt'], pred, labels)
            cm.save(f'{path[:-4]}_confusion_matrix.csv')

            # Calculate the Acc, NMI, and Novelty Detection CM
            splits[split] = {
                'accuracy': cm.accuracy(),
                'mutual_info_arithmetic': cm.mutual_info('arithmetic'),
                'mcc': cm.mcc(),
            }

            tmp_res.append([
                splits[split]['accuracy'],
                splits[split]['mutual_info_arithmetic'],
                splits[split]['mcc'],
            ])

            if args.unknowns is None:
                continue

            # Novelty Detection CM
            novelty_detect = cm.reduce(args.unknowns, 'unknown')
            novelty_detect.save(
                f'{path[:-4]}_confusion_matrix_novelty_detection.csv',
            )

            # Novelty Detection, acc, NMI, MCC
            splits[split]['novelty_detect'] = {
                'accuracy': novelty_detect.accuracy(),
                'mutual_info_arithmetic': novelty_detect.mutual_info(
                    'arithmetic',
                ),
                'mcc': novelty_detect.mcc(),
            }

            tmp_nd.append([
                splits[split]['novelty_detect']['accuracy'],
                splits[split]['novelty_detect']['mutual_info_arithmetic'],
                splits[split]['novelty_detect']['mcc'],
            ])

        # For 5 splits calculate the error/variance, ignoring last expecting it
        # to be benchmark one.
        tmp_res = np.array(tmp_res)
        splits_std = tmp_res[:-1].std(axis=0)
        stats = {
            'splits_mean': tmp_res[:-1].mean(axis=0),
            'splits_std': splits_std,
            'splits_var': tmp_res[:-1].var(axis=0),
            'splits_stderr': splits_std / np.sqrt(len(tmp_res) - 1),
        }

        if args.unknowns is not None:
            tmp_nd = np.array(tmp_nd)
            splits_nd_std = tmp_nd[:-1].std(axis=0)
            stats['novelty_detect'] = {
                'splits_mean': tmp_nd[:-1].mean(axis=0),
                'splits_std': splits_nd_std,
                'splits_var': tmp_nd[:-1].var(axis=0),
                'splits_stderr': splits_nd_std / np.sqrt(len(tmp_nd) - 1),
            }

        results[dfs.model] = {
            'splits': splits,
            'splits_stats': stats,
        }

    # Save the measurements
    with open(create_filepath(
        os.path.join(args.experiment_dir, 'results.json')
    )) as openf:
        json.dump(
            results,
            openf,
            indent=2,
            #sortkeys=True,
            cls=NumpyJSONEncoder,
        )
