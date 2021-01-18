"""Calculate the transcription measures on the data and save them."""
import glob
import json
import logging
import os
import pickle

import numpy as np

from exputils.io import parse_args, create_filepath, NumpyJSONEncoder
from exputils.data import ConfusionMatrix

from experiments.research.par_v1.grieggs import error_rates

def script_args(parser):
    parser.add_argument('experiment_dir', help='Directory of experiments.')
    #parser.add_argument('models', help='Dirs of models in experiment dir.')
    #parser.add_argument(
    #    '--unknowns',
    #    default=None,
    #    help='labels treated as unknown.',
    #)

    parser.add_argument(
        'output_path',
        help='output directory.',
        dest='output_path',
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

    unknown_char = '#'
    #output_path = os.path.join(args.experiment_dir, 'transcript_results')
    output_path = args.output_path

    known_char_dict = {"idx_to_char": {"1": " ", "2": "!", "3": "\"", "4": "#", "5": "&", "6": "'", "7": "(", "8": ")", "9": "*", "10": "+", "11": ",", "12": "-", "13": ".", "14": "/", "15": "0", "16": "1", "17": "2", "18": "3", "19": "4", "20": "5", "21": "6", "22": "7", "23": "8", "24": "9", "25": ":", "26": ";", "27": "?", "28": "A", "29": "B", "30": "C", "31": "D", "32": "E", "33": "F", "34": "G", "35": "H", "36": "I", "37": "J", "38": "K", "39": "L", "40": "M", "41": "N", "42": "O", "43": "P", "44": "Q", "45": "R", "46": "S", "47": "T", "48": "U", "49": "V", "50": "W", "51": "X", "52": "Y", "53": "Z", "54": "a", "55": "b", "56": "c", "57": "d", "58": "e", "59": "f", "60": "g", "61": "h", "62": "i", "63": "j", "64": "k", "65": "l", "66": "m", "67": "n", "68": "o", "69": "p", "70": "q", "71": "r", "72": "s", "73": "t", "74": "u", "75": "v", "76": "w", "77": "x", "78": "y", "79": "z"}, "char_to_idx": {"!": 2, " ": 1, "#": 4, "\"": 3, "'": 6, "&": 5, ")": 8, "(": 7, "+": 10, "*": 9, "-": 12, ",": 11, "/": 14, ".": 13, "1": 16, "0": 15, "3": 18, "2": 17, "5": 20, "4": 19, "7": 22, "6": 21, "9": 24, "8": 23, ";": 26, ":": 25, "?": 27, "A": 28, "C": 30, "B": 29, "E": 32, "D": 31, "G": 34, "F": 33, "I": 36, "H": 35, "K": 38, "J": 37, "M": 40, "L": 39, "O": 42, "N": 41, "Q": 44, "P": 43, "S": 46, "R": 45, "U": 48, "T": 47, "W": 50, "V": 49, "Y": 52, "X": 51, "Z": 53, "a": 54, "c": 56, "b": 55, "e": 58, "d": 57, "g": 60, "f": 59, "i": 62, "h": 61, "k": 64, "j": 63, "m": 66, "l": 65, "o": 68, "n": 67, "q": 70, "p": 69, "s": 72, "r": 71, "u": 74, "t": 73, "w": 76, "v": 75, "y": 78, "x": 77, "z": 79}}

    known_chars = set(known_char_dict['idx_to_char'].values())

    model_res = {}
    model = 'crnn'
    model_res[model] = {'folds':{}}

    for fold in range(5):
        fold_res = {}

        for split in ['train', 'val', 'test']:
            # TODO confirm this is the correct filename parsing
            pred_path = glob.glob(os.path.join(
                args.experiment_dir, f'config{fold}_{split}_preds.pkl'
            ))[0]
            gt_path = glob.glob(os.path.join(
                args.experiment_dir, f'config{fold}_{split}_gt.pkl'
            ))[0]

            # Open the ground truth and the predicted transcriptions
            with open(gt_path, 'rb') as openf:
                gt = pickle.load(openf)

            with open(pred_path, 'rb') as openf:
                pred = pickle.load(openf)

            # Check pred and gt pairing to ensure they match
            assert len(gt) == len(pred)
            assert gt.keys() == pred.keys()

            # Initialize the saved values for the split of this fold
            cer_sum = 0
            wer_sum = 0

            unique_chars = set()

            pred_nd = []
            actual_nd = []

            for key, actual in gt.items():
                # Calculate the transcription results
                cer = error_rates.cer(actual, pred[key])
                wer = error_rates.wer(actual, pred[key])

                cer_sum += cer
                wer_sum += wer

                actual_char_set = set(actual)
                unique_chars |= actual_char_set

                # Novelty predicted when novel/unknown char '#' in pred
                pred_nd.append(
                    'unknown' if unknown_char in pred[key] else 'known'
                )
                # novelty exists when novel char in gt
                if any([v not in known_chars for v in actual_char_set]):
                    actual_nd.append('unknown')
                else:
                    actual_nd.append('known')

            # Calculate the Novel Character Detection confusion matrix
            nd_cm = ConfusionMatrix(
                actual_nd,
                pred_nd,
                labels=['known', 'unknown'],
            )
            nd_cm.save(os.path.join(
                output_path,
                f'f{fold}-{split}_novel_char_detect_per_line_confusion_matrix.csv',
            ))

            cer = cer_sum / len(gt)
            wer = wer_sum / len(gt)

            fold_res[split] = {
                'cer': cer,
                'wer': wer,
                'char_acc': 1 - cer,
                'word_acc': 1 - wer,
                'novelty_detect': {
                    'accuracy': nd_cm.accuracy(),
                    'mutual_info_arithmetic': nd_cm.mutual_information(
                        'arithmetic',
                    ),
                    'mcc': nd_cm.mcc(),
                },
                'chars': {
                    'unique_chars': unique_chars,
                    'unknown_chars': unique_chars - known_chars,
                },
            }
        model_res[model]['folds'][f'fold_{fold}'] = fold_res

    # For 5 splits calculate the error/variance, ignoring benchmark set.
    #folds = [
    #    v for k, v in model_res[model]['folds'].items() if 'fold_' in k
    #]

    model_res[model]['folds_stats'] = {}

    for dsplit in ('train', 'val', 'test'):
        cers = []
        wers = []
        char_accs = []
        word_accs = []

        nd_accs = []
        nd_nmis = []
        nd_mccs = []

        for fold in model_res[model]['folds'].values():
            cers.append(fold[dsplit]['cer'])
            wers.append(fold[dsplit]['wer'])
            char_accs.append(fold[dsplit]['char_accs'])
            word_accs.append(fold[dsplit]['word_accs'])

            nd_accs.append(fold[dsplit]['novelty_detect']['accuracy'])
            nd_nmis.append(
                fold[dsplit]['novelty_detect']['mutual_info_arithmetic']
            )
            nd_mccs.append(fold[dsplit]['novelty_detect']['mcc'])

        model_res[model]['folds_stats'][dsplit] = {
            'cer': stat(cers),
            'wer': stat(wers),
            'char_acc': stat(char_accs),
            'word_acc': stat(word_accs),
            'novelty_detect': {
                'accuracy': stat(nd_mccs),
                'mutual_info_arithmetic': stat(nd_nmis),
                'mcc': stat(nd_mccs),
            },
        }

    # Save the measurements
    with open(create_filepath(
        os.path.join(output_path, 'results.json')
    ), 'w') as openf:
        json.dump(
            model_res,
            openf,
            indent=2,
            cls=NumpyJSONEncoder,
        )
