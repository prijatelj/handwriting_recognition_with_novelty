"""Script for splitting the given data json into multiple data JSONs for each
train and test set per fold.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

import exputils.io


def parse_args(parser):
    parser.add_argument(
        'labels_filepath',
        help='The filepath to the PAR label CSV.',
    )

    parser.add_argument(
        'output_dir',
        default=None,
        help='The directory filepath used to save the resulting TSVs.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=4816,
        help='The seed used to initialize the Kfold CV split.',
    )

    parser.add_argument(
        '-d',
        '--delimiter',
        default='\t',
        help='The delimiter used by the PAR labels CSV.',
    )

    parser.add_argument(
        '--stratified_label_set',
        default='writer_id',
        help='The label set id (column) to stratifiy about for the KFold CV.',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If true, creates a new directory if one already exists.',
    )


if __name__ == '__main__':
    args = exputils.io.parse_args(
        ['kfold'],
        parse_args,
        description='Splits the given labels TSV into multiple',
    )

    # Load the given data tsv
    labels = pd.read_csv(args.labels_filepath, sep=args.delimiter)

    # Data index splitting: Use kfold (stratified if told to) to split the data
    #if args.kfold_cv.stratified:
    if False:
        # If there are writers with less samples than folds, temporarily remove
        # them and randomly add them to a fold.
        idx, counts = np.unique(
            labels[args.stratified_label_set],
            return_counts=True,
        )

        ids = idx[np.where(counts < args.kfold_cv.kfolds)]

        if len(ids) > 0:
            labels['index'] = labels.index
            writers = labels.set_index(args.stratified_label_set)
            under_sampled = writers.loc[ids]['index']
            del labels['index']

            # Create the splits of these under sampled for each fold
            np.random.seed(args.seed)
            under_fold_indices = list(KFold(
                args.kfold_cv.kfolds,
                shuffle=False,
                random_state=args.seed,
            ).split(
                under_sampled.array,
                labels[args.stratified_label_set][under_sampled.array],
            ))

            # Remove the under sampled writers from the labels
            labels.drop(index=under_sampled.array)

            fold_indices = list(StratifiedKFold(
                args.kfold_cv.kfolds,
                shuffle=False,
                random_state=args.seed,
            ).split(labels.index, labels[args.stratified_label_set]))

            # Seed the generator and begin shuffling
            np.random.seed(args.seed)

            # Concat train and test indices together per fold
            for i in range(args.kfold_cv.kfolds):
                # train
                train_fold_indices = np.concatenate((
                    fold_indices[i][0],
                    under_fold_indices[i][0],
                ))

                print

                # test
                test_fold_indices = np.concatenate((
                    fold_indices[i][1],
                    under_fold_indices[i][1],
                ))

                # Shuffle
                if args.kfold_cv.shuffle:
                    np.random.shuffle(train_fold_indices)
                    np.random.shuffle(test_fold_indices)

                fold_indices[i] = [train_fold_indices, test_fold_indices]

        else:
            # Split the data into stratified folds, preserving percentage of
            # samples for each class among the folds.
            fold_indices = StratifiedKFold(
                args.kfold_cv.kfolds,
                shuffle=args.kfold_cv.shuffle,
                random_state=args.seed,
            ).split(labels.index, labels[args.stratified_label_set])
    else:
        fold_indices = KFold(
            args.kfold_cv.kfolds,
            shuffle=args.kfold_cv.shuffle,
            random_state=args.seed,
        ).split(labels.index, labels[args.stratified_label_set])

    # Ensure the output directory exists
    output_dir = exputils.io.create_dirs(args.output_dir)

    # For each fold, save the train and test data TSVs.
    for i, (train_fold, test_fold) in enumerate(fold_indices):
        labels.iloc[train_fold].to_csv(
            os.path.join(
                output_dir,
                f'{args.kfold_cv.kfolds}folds_train-{i}.tsv',
            ),
            sep=args.delimiter,
            index=False,
        )

        labels.iloc[test_fold].to_csv(
            os.path.join(
                output_dir,
                f'{args.kfold_cv.kfolds}folds_test-{i}.tsv',
            ),
            sep=args.delimiter,
            index=False,
        )
