"""Label managment for transcription and other related handwriting recognitoin
tasks, e.g. style.
"""
from exputils.data.labels import NominalDataEncoder, load_label_set


class CharEncoder(NominalDataEncoder):
    """Temporary hotfix for bringing all the label data together in one place.
    Wraps NominalDataEncoder to include character specific things.
    """
    def __init__(self, blank_idx, space_char, unknown_idx, *args, **kwargs):
        super(CharEncoder, self).__init__(*args, **kwargs)

        # NOTE CRNN expects 0 idx by default, any unique label
        self.blank_idx = blank_idx

        # NOTE CRNN expects ' ', any idx (def: 1)
        self.space_char = space_char

        # NOTE CRNN expects this to add to end of labels, any unique label.
        self.unknown_idx = unknown_idx

    # TODO copy/modify/replace str2label and label2str, etc from string_utils
    # and put them here within the character encoder!
    # NOTE if you turn the str into a list of chars, [en/de]code will return
    # numpy arrays and function as expected... Just need to type cast np.uint32

    # TODO posisbly include error_rates here or edit dist method if dependent
    # on character encoding: e.g. blank, space char, or unknown idx.

    @property
    def blank_char(self):
        return self.encoder.inverse[self.blank_idx]

    @property
    def space_idx(self):
        return self.encoder[self.space_char]

    @property
    def unknown_char(self):
        return self.encoder.inverse[self.unknown_idx]

    @staticmethod
    def load(filepath, blank_idx, space_char, unknown_idx):
        """Loads the label set and creates char encoder"""
        nde = NominalDataEncoder.load(filepath)

        # TODO build CharEncoder s.t. it can simply copy the parts of the given
        # NDE
        return CharEncoder(blank, space_char, unknown_idx, list(nde.encoder))


def load_char_encoder(filepath, blank, space_char, unknown_idx):
    """Loads the label set and creates char encoder"""
    nde = load_label_set(filepath)

    return CharEncoder(blank, space_char, unknown_idx, list(nde.encoder))


@dataclass
class TranscriptResults:
    """Contains everything for the dataset handling in one place."""
    char_error_rate: float
    word_error_rate: float
