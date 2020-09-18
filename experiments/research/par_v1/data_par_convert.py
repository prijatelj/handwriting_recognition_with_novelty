"""Convert the given PAR files into the desired data TSVs"""
import argparse
import csv
import os

import pandas as pd

import exputils

def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine PAR data csvs into one tsv.',
    )
    parser.add_argument(
        'labels_filepath',
        help='The filepath to the PAR label CSV.',
    )

    parser.add_argument(
        '-t',
        '--transcripts',
        default=None,
        help='The filepath to the transcript text CSV.',
    )

    parser.add_argument(
        '-o',
        '--output_filepath',
        default=None,
        help='The filepath used to save the resulting CSV.',
    )

    parser.add_argument(
        '-a',
        '--append_filepath',
        default=None,
        help='The filepath to be appended to all of the image filepaths.',
    )

    parser.add_argument(
        '-d',
        '--delimiter',
        default=',',
        help='The delimiter used by the PAR labels CSV.',
    )

    parser.add_argument(
        '-D',
        '--transcript_delimiter',
        default='|',
        help='The delimiter used.',
    )

    parser.add_argument(
        'c',
        '--create_char_set',
        action='store_true',
        help='Create the label encoding csv from unique chars in transcript',
    )

    parser.add_argument(
        '--char_set_as_json',
        action='store_true',
        help='Save char set as JSON.',
    )

    parser.add_argument(
        '--save_delimiter',
        default='\t',
        help='The delimiter used in resulting TSV, default being tabs.',
    )

    args = parser.parse_args()

    if args.output_filepath is None:
        args.output_filepath = './PAR_converted_labels.csv'

    exputils.io.create_filepath(args.output_filepath)

    return args

# TODO arg parser to load the PAR files from given filepaths
args = parse_args()

# TODO load the given files: labels, transcripts
# Load labels
labels = pd.read_csv(args.labels_filepath, sep=args.delimiter, header=True)

# TODO append to image filepaths if necessary
if args.append_filepath is not None:


if os.path.isfile(args.transcripts_filepath):
    # TODO Load the PAR transcripts file

    # TODO Parse the PAR transcripts file, forming the transcript text as 1 str

    # TODO Obtain the unique characters

    # Save the resulting char set as a CSV

# TODO Save the resulting labels TSV, handling csv quoting appropriately
