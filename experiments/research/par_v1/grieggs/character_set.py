import sys
import json
import os
from collections import defaultdict

def load_label_set(filepath, delimiter=None, increment_enc=None):
    """Loads the given file and reads the labels in. Expects label per line.

    Parameters
    ----------
    filepath : str
        The filepath to the file containing the labels
    delmiter : str, optional
        The delimiter character if the file contains provided encodings per
        label. Always assumes one label per line. Will assume first column is
        the original label to be encoded to the provided encoding when
        delimiter is not None.
    increment_enc : int, optional
        Optional incrementation to the encoding values
    """
    if increment_enc is not None:
        raise NotImplementedError('incrementing the encoding integer values.')

    if delimiter is None:
        with open(filepath, 'r') as openf:
            nd_enc = openf.read().splitlines()

        idx_to_char = {}
        char_to_idx = {}
        for i, label in enumerate(nd_enc):
            idx_to_char[label] = i
            char_to_idx[i] = label

        return idx_to_char, char_to_idx

    # TODO load as csv or tsv. And actually finish in exputils


def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    if sys.hexversion >= 0x3000000:
        for k,v in char_set['idx_to_char'].items():
            idx_to_char[int(k)] = v
    else:
        for k,v in char_set['idx_to_char'].iteritems():
            idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']

if __name__ == "__main__":
    character_set_path = sys.argv[-1]
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)
    for i in range(1, len(sys.argv)-1):
        data_file = sys.argv[i]
        with open(data_file) as f:
            data = json.load(f)

        cnt = 1 # this is important that this starts at 1 not 0
        for data_item in data:
            for c in data_item.get('gt', ""):
                if c not in out_char_to_idx:
                    out_char_to_idx[c] = cnt
                    out_idx_to_char[cnt] = c
                    cnt += 1
                char_freq[c] += 1

    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i+1
        out_idx_to_char2[i+1] = c

    output_data = {
        "char_to_idx": out_char_to_idx2,
        "idx_to_char": out_idx_to_char2
    }

    for k,v in sorted(char_freq.iteritems(), key=lambda x: x[1]):
        print(k, v)

    print("Size:", len(output_data['char_to_idx']))

    with open(character_set_path, 'w') as outfile:
        json.dump(output_data, outfile)
