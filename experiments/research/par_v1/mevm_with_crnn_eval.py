"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
# Python default packages
import logging

# 3rd party packages
import numpy as np
from ruamel.yaml import YAML
import torch

# External packages but within project
from evm_based_novelty_detector.MultipleEVM import MultipleEVM as MEVM
import exputils.io

from experiments.research.par_v1 import crnn_script, crnn_data
from experiments.research.par_v1 import mevm_with_crnn


def custom_args(parser):
    mevm_with_crnn.script_args(parser)

    parser.add_argument(
        '--mevm_path',
        default=None,
        help='The path to the trained MEVM state.',
    )


def main():
    args = exputils.io.parse_args(custom_args=custom_args)

    if args.random_seed:
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True

        logging.info('Random seed = %d', args.random_seed)

    with open(args.config_path) as openf:
        config = YAML(typ='safe').load(openf)

    # Load the data
    train_dataloader, test_dataloader, char_enc = crnn_data.load_data(
        config,
        args.col_chars_path,
    )

    # Load CRNN
    crnn, dtype = crnn_data.init_CRNN(config)

    # Load MEVM nominal encoder
    nominal_enc = mevm_with_crnn.load_col_chars(
        char_enc,
        args.col_chars_path,
        args.blank_repr_div,
        args.unknown_char_extra_neg,
    )[0]


    # Init MEVM from config
    mevm = MEVM(device='cpu', **config['model']['mevm']['init'])

    # load MEVM
    mevm.load(args.mevm_path)

    # Eval
    if 'train' in args.eval:
        # TODO make sure the data loader is not shuffling! Error ow.
        # TODO warn if the data loader uses augmentation
        results = mevm_with_crnn.eval_crnn_mevm(
            crnn,
            mevm,
            train_dataloader,
            char_enc,
            nominal_enc,
            dtype,
            layer=args.layer,
            decode=args.decode,
            threshold=args.unknown_threshold,
        )

        logging.info(
            'train eval performance: CER: %f; WER: %f',
            results.char_error_rate,
            results.word_error_rate,
        )

    if 'test' in args.eval:
        # TODO make sure the data loader is not shuffling! Error ow.
        # TODO warn if the data loader uses augmentation
        results = mevm_with_crnn.eval_crnn_mevm(
            crnn,
            mevm,
            test_dataloader,
            char_enc,
            nominal_enc,
            dtype,
            layer=args.layer,
            decode=args.decode,
            threshold=args.unknown_threshold,
        )

        logging.info(
            'test eval performance: CER: %f; WER: %f',
            results.char_error_rate,
            results.word_error_rate,
        )


if __name__ == "__main__":
    main()
