"""Calculate the measures on the data and save the confusion matrices."""
from collections import namedtuple
import glob
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

    parser.add_argument(
        '--train_suffix',
        default='*_repr_aug.csv',
        help='Ending filename of train csv',
    )

    parser.add_argument(
        '--val_suffix',
        default='*_eval/val.csv',
        help='Ending filename of val csv',
    )

    parser.add_argument(
        '--test_suffix',
        default='*_eval/test.csv',
        help='Ending filename of test csv',
    )


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

    logging.debug('fpr = %f; fnr = %f', fpr, fnr)

    return (fpr - fnr)**2

    #tpr = cm.true_rate(unk_idx)
    #return -(tpr + fpr)


def get_cm(actual, probs, labels, threshold, unknowns, unk_idx, base_path):
    pred = probs.argmax(1)
    pred[probs[np.arange(probs.shape[0]), pred] < threshold] = unk_idx
    pred = labels[pred]

    cm = ConfusionMatrix(actual, pred, labels)
    if base_path is not None:
        cm.save(f'{base_path}_confusion_matrix_thresh-{threshold}.csv')

    # Reduce the known unknowns to unknown for the measures!
    cm = cm.reduce(args.unknowns, 'unknown')

    # Novelty Detection CM
    novelty_detect_cm = cm.reduce(
        unknowns,
        'known',
        inverse=True,
    )
    novelty_detect_cm.save(
        f'{base_path}_confusion_matrix_thresh-{threshold}_novelty_detection.csv',
    )

    return cm, novelty_detect_cm


def stat(arr):
    std = np.std(arr)
    return {
        'mean': np.mean(arr),
        'std': std,
        'var': np.var(arr),
        'stderr': std / np.sqrt(len(arr)),
    }


if __name__ == '__main__':
    args = parse_args(custom_args=script_args)

    args.models = args.models.split(' ')
    if args.unknowns is not None:
        args.unknowns = args.unknowns.split(' ')
        if 'unknown' not in args.unknowns:
            args.unknowns = args.unknowns + ['unknown']
    elif 'writer_id' not in args.experiment_dir:
        raise NotImplementedError('need unknowns')

    model_res = {}
    for model in args.models:
        model_path = os.path.join(args.experiment_dir, model)
        model_res[model] = {'folds':{}}

        for fold_path in glob.iglob(os.path.join(model_path, '*')):
            fold_res = {}
            #fold_path = os.path.join(model_path, fold)

            train_path = glob.glob(
                os.path.join(fold_path, args.train_suffix)
            )[0]
            train_df = pd.read_csv(train_path)

            val_path = glob.glob(
                os.path.join(fold_path, args.val_suffix)
            )[0]
            val_df = pd.read_csv(val_path)

            assert (train_df.columns == val_df.columns).all

            # Load test probs
            test_path = glob.glob(os.path.join(fold_path, args.test_suffix))[0]
            test_df = pd.read_csv(test_path)

            all_gt_labels = (
                set(train_df['gt'])
                | set(val_df['gt'])
                | set(test_df['gt'])
            )

            # Get labels and the index of them
            missed_labels = list(all_gt_labels - set(train_df.columns[2:]))
            labels = np.array(list(train_df.columns[2:]) + missed_labels)

            # Set Unknowns
            if 'writer_id' not in args.experiment_dir:
                # If writer id, then each fold has its own set of unknown
                # writers in val and test
                unknowns = list(set(missed_labels) + {'unknown'})
            else:
                unknowns = args.unkowns

            unk_idx = np.where(labels == 'unknown')[0][0]

            # Find the optimal threshold based on train and val
            if args.min_opt is None:
                logging.info(' '.join([
                    'Not optimizing on this dataset, simply applying the',
                    'threshold.',
                ]))

                threshold = args.init_thresh
            elif args.min_opt == 'linspace':
                probs = np.concatenate((
                    train_df[train_df.columns[2:]].values,
                    val_df[val_df.columns[2:]].values,
                ))

                actuals = np.concatenate((
                    train_df['gt'].values,
                    val_df['gt'].values,
                ))

                threshold = None

                min_val = np.inf
                for thresh in np.linspace(0, 1, 81):
                    val = crossover_error_rate_opt(
                        [thresh],
                        actuals,
                        probs,
                        labels,
                        unknowns,
                        unk_idx,
                    )

                    logging.debug('thres = %f; val = %f', thresh, val)

                    if val < min_val:
                        min_val = val
                        threshold = thresh
            else:
                raise NotImplementedError('does not work well for discrete.')
                probs = np.concatenate((
                    train_df[train_df.columns[2:]].values,
                    val_df[val_df.columns[2:]].values,
                ))

                actuals = np.concatenate((
                    train_df['gt'].values,
                    val_df['gt'].values,
                ))

                opt_result = minimize(
                    crossover_error_rate_opt,
                    [args.init_thresh],
                    (actuals, probs, labels, unknowns, unk_idx),
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

            logging.info(
                'Threshold is `%f` for fold `%s`',
                threshold,
                fold_path,
            )

            fold_res['threshold'] = threshold

            # Save measures for train, val, and test
            for split, df, path in (
                ('train', train_df, train_path),
                ('val', val_df, val_path),
                ('test', test_df, test_path),
            ):
                cm, nd_cm = get_cm(
                    df['gt'],
                    df[df.columns[2:]].values,
                    labels,
                    threshold,
                    unknowns,
                    unk_idx,
                    path[:-4],
                )

                fold_res[split] = {
                    'accuracy': cm.accuracy(),
                    'mutual_info_arithmetic': cm.mutual_information(
                        'arithmetic',
                    ),
                    'mcc': cm.mcc(),
                    'novelty_detect': {
                        'accuracy': nd_cm.accuracy(),
                        'mutual_info_arithmetic': nd_cm.mutual_information(
                            'arithmetic',
                        ),
                        'mcc': nd_cm.mcc(),
                    }
                }

            model_res[model]['folds'][os.path.basename(fold_path)] = fold_res

        # For 5 splits calculate the error/variance, ignoring benchmark set.
        folds = [
            v for k, v in model_res[model]['folds'].items() if 'split_' in k
        ]

        model_res[model]['folds_stats'] = {}

        for dsplit in ('train', 'val', 'test'):
            accs = []
            nmis = []
            mccs = []

            nd_accs = []
            nd_nmis = []
            nd_mccs = []

            for fold in folds:
                accs.append(fold[dsplit]['accuracy'])
                nmis.append(fold[dsplit]['mutual_info_arithmetic'])
                mccs.append(fold[dsplit]['mcc'])

                nd_accs.append(fold[dsplit]['novelty_detect']['accuracy'])
                nd_nmis.append(
                    fold[dsplit]['novelty_detect']['mutual_info_arithmetic']
                )
                nd_mccs.append(fold[dsplit]['novelty_detect']['mcc'])

            model_res[model]['folds_stats'][dsplit] = {
                'accuracy': stat(accs),
                'mutual_info_arithmetic': stat(nmis),
                'mcc': stat(mccs),
                'novelty_detect': {
                    'accuracy': stat(nd_mccs),
                    'mutual_info_arithmetic': stat(nd_nmis),
                    'mcc': stat(nd_mccs),
                },
            }

    # Save the measurements
    with open(create_filepath(
        os.path.join(args.experiment_dir, 'results.json')
    ), 'w') as openf:
        json.dump(
            model_res,
            openf,
            indent=2,
            #sortkeys=True,
            cls=NumpyJSONEncoder,
        )
