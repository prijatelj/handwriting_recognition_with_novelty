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
        '--transcript_csv_delimiter',
        default='\t',
        help='The delimiter used.',
    )
    parser.add_argument(
        '--transcript_text_delimiter',
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


    return args

# arg parser to load the PAR files from given filepaths
args = parse_args()

# Load the given files: labels, transcripts
# Load labels
labels = pd.read_csv(args.labels_filepath, sep=args.delimiter, header=True)

# Append to image filepaths if necessary
if args.append_filepath is not None:
    for i in range(len(labels)):
        labels['file'] = os.path.join(args.append_filepath, labels['file'])

if os.path.isfile(args.transcripts_filepath):
    # Load the PAR transcripts file
    with open(args.transcripts_filepath) as transcript_csv:
        csv_reader = csv.reader(
            transcript_csv,
            delimiter=args.transcript_csv_delimiter,
            quoting=csv.QUOTE_NONE,
        )

        # Parse the PAR transcripts file, forming the transcript text as 1 str
        transcript_map = dict()
        if args.create_char_set:
            unique_characters = set()

        for row in csv_reader:
            if row[0] in transcript_map:
                raise KeyError('Duplicate key exists in transcripts csv!')

            # Attempt to recover the acutal sentence string where possible
            if ',' in row[1]:
                row[1] = row[1].replace('|,', ',')
            if '.' in row[1]:
                row[1] = row[1].replace('|.', '.')
            if '?' in row[1]:
                row[1] = row[1].replace('|?', '?')
            if '!' in row[1]:
                row[1] = row[1].replace('|!', '!')
            if ':' in row[1]:
                row[1] = row[1].replace('|:', ':')

            # Handle the ambiguous cases by not allowing any white space to
            # surround them because white space is a character in itself.
            # NOTE currently i let ' and " and - as separate words, meaning
            # that white space will surround them.

            transcript_map[row[0]] = row[1].replace('|', ' ')

            # Obtain the unique characters
            if args.create_char_set:
                unique_characters |= set(transcript_map[row[0]])

            # Add transcript line to the labels csv

    # Check if any image in labels is missing a transcript

    # Save the resulting char set as a CSV
    if args.create_char_set:
        # Create character set file at same directory as the output labels file
        char_set_filepath = args.output_filepath.rpartition(os.path.sep)

        if char_set_filepath[0]:
            char_set_filepath = os.path.join(
                char_set_filepath[0],
                'char_set.csv',
            )
        else:
            char_set_filepath = 'char_set.csv'

        with open(exputils.io.create_filepath(char_set_filepath), 'w') as f:
            # Create a simple "csv" file where all unique characters are on
            # their own line. The idx is then able to be inferred by their
            # order.
            for character in unique_characters:
                f.write(f'{character}\n')


# Save the resulting labels TSV, handling csv quoting appropriately
labels.to_csv(
    exputils.io.create_filepath(args.output_filepath),
    sep=args.save_delimiter,
    index=False,
)
