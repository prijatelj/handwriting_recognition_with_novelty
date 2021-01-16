"""Calculate the measures on the data and save the confusion matrices."""
from collections import namedtuple
from glob import glob
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix

from exputils.io import parse_args, create_filepath, NumpyJSONEncoder
from exputils.data import ConfusionMatrix
#from exputils.data.labels import NominalDataEncoder

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

    parser.add_argument(
        '--init_thresh',
        default=0.5,
        type=float,
        help='Initial threshold of unknowns.',
    )

    parser.add_argument(
        '--min_opt',
        default=None,
        choices=['linspace', 'TNC', 'L-BFGS-B', 'SLSQP', 'Powell'],
        help='labels treated as unknown.',
    )


def get_dfs(experiment_dir, models):
    paths = []
    dfs = []
    Probs = namedtuple('Probs', ['model', 'train', 'val', 'test'])
    DFP = namedtuple('DFPath', ['split', 'path', 'df'])

    for model in models:
        path = os.path.join(experiment_dir, model)
        if os.path.isdir(path):
            train = [
                g for g in glob(f'{path}/*/*[!_points].csv')
                if 'confusion' not in g
            ]
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

    # TODO couple train and val together so the thresh can be found on them.

    return dfs


def crossover_error_rate_opt(
    threshold,
    actuals,
    probs,
    labels,
    unknowns,
    unk_idx,
):
    """Calculates the squared difference between error rates."""
    # TODO apply thresholding method: max prob < thresh then unknown
    argmax = probs.argmax(1)
    argmax[probs[np.arange(probs.shape[0]), argmax] < threshold[0]] = unk_idx

    cm = ConfusionMatrix(
        actuals,
        labels[argmax],
        labels,
    ).reduce(unknowns, 'unknown')
    fpr, fnr = cm.false_rates(unk_idx)

    return (fpr - fnr)**2

    #tpr = cm.true_rate(unk_idx)
    #return -(tpr + fpr)


if __name__ == '__main__':
    args = parse_args(custom_args=script_args)

    args.models = args.models.split(' ')
    if args.unknowns is not None:
        args.unknowns = args.unknowns.split(' ')
        if 'unknown' not in args.unknowns:
            args.unknowns = args.unknowns + ['unknown']

    # Load the probs csvs
    prob_dfs = get_dfs(args.experiment_dir, args.models)

    # TODO load two at a time: train and val and assess threshold given them

    # Create a confusion matrix for each probs csv and save
    results = {}
    for dfs in prob_dfs:
        splits = {}

        # Skip first since that is the model str.
        for split, path, df in dfs[1:]:
            tmp_res = []
            tmp_nd = []

            splits[split] = {}

            for i, dat in enumerate(df):
                probs = dat[dat.columns[2:]].values

                # Get labels and the index of them
                missed_labels = list(set(dat['gt']) - set(dat.columns[2:]))
                labels = np.array(list(dat.columns[2:]) + missed_labels)

                unk_idx = np.where(labels == 'unknown')[0][0]

                if args.min_opt is None:
                    logging.info(' '.join([
                        'Not optimizing on this dataset, simply applying the',
                        'threshold.',
                    ]))
                    threshold = args.init_thresh
                elif args.min_opt == 'linspace':
                    threshold = None

                    min_val = np.inf
                    for thresh in np.linspace(0, 1, 81):
                        val = crossover_error_rate_opt(
                            [thresh],
                            dat['gt'].values,
                            probs,
                            labels,
                            args.unknowns,
                            unk_idx,
                        )

                        logging.info('thres = %f; val = %f', thresh, val)

                        if val < min_val:
                            min_val = val
                            threshold = thresh
                else:
                    opt_result = minimize(
                        crossover_error_rate_opt,
                        [args.init_thresh],
                        (dat['gt'].values, probs, labels, args.unknowns, unk_idx),
                        method=args.min_opt,
                        bounds=[(0.0, 1.0)],
                    )

                    if not opt_result.success:
                        raise ValueError(' '.join([
                            'Unsuccessful threshold optimization! message:',
                            f'{opt_result.message}',
                        ]))
                    else:
                        logging.debug('opt results: %s', opt_result)

                    threshold = opt_result.x[0]

                logging.info('Threshold is `%f` for `%s`', threshold, path[i])

                pred = probs.argmax(1)
                pred[probs[np.arange(probs.shape[0]), pred] < threshold] = unk_idx
                pred = list(dat.columns[pred + 2])

                cm = ConfusionMatrix(dat['gt'], pred, labels)
                cm.save(
                    f'{path[i][:-4]}_confusion_matrix_thresh-{threshold}.csv',
                )

                # Calculate the Acc, NMI, and Novelty Detection CM

                # Reduce the known unknowns to unknown for the measures!
                cm = cm.reduce(args.unknowns, 'unknown')

                splits[split][i] = {
                    'accuracy': cm.accuracy(),
                    'mutual_info_arithmetic': cm.mutual_information(
                        'arithmetic',
                    ),
                    'mcc': cm.mcc(),
                }

                tmp_res.append([
                    splits[split][i]['accuracy'],
                    splits[split][i]['mutual_info_arithmetic'],
                    splits[split][i]['mcc'],
                ])

                if args.unknowns is None:
                    continue

                # Novelty Detection CM
                novelty_detect = cm.reduce(
                    args.unknowns,
                    'known',
                    inverse=True,
                )
                novelty_detect.save(
                    f'{path[i][:-4]}_confusion_matrix_novelty_detection.csv',
                )

                # Novelty Detection, acc, NMI, MCC
                splits[split][i]['novelty_detect'] = {
                    'accuracy': novelty_detect.accuracy(),
                    'mutual_info_arithmetic': novelty_detect.mutual_information(
                        'arithmetic',
                    ),
                    'mcc': novelty_detect.mcc(),
                }

                tmp_nd.append([
                    splits[split][i]['novelty_detect']['accuracy'],
                    splits[split][i]['novelty_detect']['mutual_info_arithmetic'],
                    splits[split][i]['novelty_detect']['mcc'],
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
    ), 'w') as openf:
        json.dump(
            results,
            openf,
            indent=2,
            #sortkeys=True,
            cls=NumpyJSONEncoder,
        )
