"""Convert the given PAR files into the desired data TSVs"""
import argparse
from collections import Counter
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
        '--transcripts_filepath',
        default=None,
        help='The filepath to the transcript text CSV.',
    )

    parser.add_argument(
        '-r',
        '--repr_filepath',
        default=None,
        help='The filepath to the representation CSV.',
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
        help='The filepath prefix to be appended to all image filepaths.',
    )

    parser.add_argument(
        '-S',
        '--append_transcript_suffix',
        default=None,
        help='Adds the suffix to the image filename in the transcript file.',
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
        '-c',
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
        '--recover_whitespace',
        action='store_true',
        help='Replace word delimiter with by inferring whitespace.',
    )

    parser.add_argument(
        '--do_not_replace_separator',
        action='store_false',
        help='Do not replace word delimiter with whitespace.',
        dest='replace_separator',
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
labels = pd.read_csv(
    args.labels_filepath,
    sep=args.delimiter,
    quoting=csv.QUOTE_NONE,
)


unique_files = len(set(labels['file']))
if len(set(labels['file'])) != len(labels.index):
    raise KeyError(' '.join([
        'The number of unique files does not equal the number of lines in the',
        f'labels CSV! CSV lines = {len(labels.index)}; Unique filenames =',
        f'{unique_files}',
    ]))
del unique_files

# Add representation columns if any
if os.path.isfile(args.repr_filepath):
    with open(args.repr_filepath) as repr_csv:
        csv_reader = csv.reader(
            repr_csv,
            delimiter=args.delimiter,
            quoting=csv.QUOTE_NONE,
        )

        # Skip expected header
        next(csv_reader)

        # Add the provided repr columns
        labels['bg_antique_provided'] = 0
        labels['noise_provided'] = 0

        file_counter = Counter()

        for row in csv_reader:
            if row[0] in file_counter and file_counter[row[0]] >= 2:
                raise KeyError(
                    f'Over 2 duplicates of a filename in repr csv! {row[0]}',
                )
            file_counter[row[0]] += 1

            # Check if the writer matches
            if str(labels['writer_id'][row[0]]) != row[1]:
                raise ValueError(' '.join([
                    'writer_id for file in repr csv does not match the labels',
                    f'csv! writer_id in labels: {labels["writer_id"][row[0]]}',
                    f'writer_id in repr csv: {row[1]}',
                ]))

            # Add the repr provided information. Dupes expected, thus |=
            labels['bg_antique_provided'][row[0]] |= int(row[2])
            labels['noise_provided'][row[0]] |= int(row[3])

# Add transcripts if any
if os.path.isfile(args.transcripts_filepath):
    # Load the PAR transcripts file
    with open(args.transcripts_filepath) as transcript_csv:
        csv_reader = csv.reader(
            transcript_csv,
            delimiter=args.transcript_csv_delimiter,
            quoting=csv.QUOTE_NONE,
        )

        if not os.path.isfile(args.repr_filepath):
            # Set the index to the unique image filenames for ease of mapping
            # transcripts
            labels.index = labels['file']

        # Set the index to the unique image filenames for ease of mapping
        # transcripts
        labels.index = labels['file']

        # Add the transcript column to the labels
        labels['transcript'] = pd.NA

        # Parse the PAR transcripts file, forming the transcript text as 1 str
        file_set = set()
        if args.create_char_set:
            unique_characters = set()

        for row in csv_reader:
            if args.append_transcript_suffix is not None:
                row[0] = row[0] + args.append_transcript_suffix

            if row[0] in file_set:
                raise KeyError('Duplicate filename exists in transcripts csv!')

            if args.replace_separator:
                # Replaces the expected pipe separator with a whitespace to
                # match CRNN expectations.
                row[1] = row[1].replace('|', ' ')
            elif args.recover_whitespace:
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

                # Final reintroduction of whitespace into the given transcription
                row[1] = row[1].replace('|', ' ')

            file_set.add(row[0])

            # Obtain the unique characters
            if args.create_char_set:
                unique_characters |= set(row[1])

            # Find row index in labels given the filename & add transcript
            labels['transcript'][row[0]] = row[1]

    # Check if any image in labels is missing a transcript
    na_transcript = pd.isna(labels['transcript'])
    if na_transcript.any():
        raise KeyError(' '.join([
            'Missing transcriptions for some images! Total:',
            f'{na_transcript.sum()}.The following do not',
            f'have a provided transcript:\n{labels["file"][na_transcript]}',
        ]))
    del na_transcript

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
            for character in sorted(unique_characters):
                f.write(f'{character}\n')

# Append to image filepaths if necessary, after adding the transcriptions
if args.append_filepath is not None:
    #for i in range(len(labels)):
    labels['file'] = args.append_filepath + labels['file']

# Save the resulting labels TSV, handling csv quoting appropriately
labels.to_csv(
    exputils.io.create_filepath(args.output_filepath),
    sep=args.save_delimiter,
    index=False,
    quoting=csv.QUOTE_NONE,
)
