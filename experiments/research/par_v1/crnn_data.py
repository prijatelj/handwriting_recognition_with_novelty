"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
from __future__ import print_function

# Python built-in packages
from copy import deepcopy
from dataclasses import dataclass
import logging

# 3rd party packages
import editdistance
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset

# Internal package modules
from hwr_novelty.models.crnn import CRNN

from exputils.data import ConfusionMatrix
from exputils.data.labels import NominalDataEncoder, load_label_set
from experiments.research.par_v1.grieggs import (
    character_set,
    error_rates,
    hw_dataset,
    string_utils,
)


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


def load_char_encoder(filepath, blank, space_char, unknown_idx):
    """Loads the label set and creates char encoder"""
    nde = load_label_set(filepath)

    return CharEncoder(blank, space_char, unknown_idx, list(nde.encoder))


@dataclass
class DataSet:
    """Contains everything for the dataset handling in one place."""
    label_encoder: CharEncoder
    train_dataset: hw_dataset.HwDataset = None
    train_dataloader: DataLoader = None
    val_dataset: hw_dataset.HwDataset = None
    val_dataloader: DataLoader= None
    test_dataset: hw_dataset.HwDataset = None
    test_dataloader: DataLoader = None


@dataclass
class TranscriptResults:
    """Contains everything for the dataset handling in one place."""
    char_error_rate: float
    word_error_rate: float
    char_confusion_mat: np.ndarray = None


def eval_transcription_logits(
    texts,
    logits,
    label_encoder,
    decode='naive',
):
    """Evaluates the given predicted transcriptions to expected texts where the
    predictions are given as a probability vector per character.

    Parameters
    ----------
    preds : np.ndarray
        array of shape [samples, timesteps, characters], where samples is the
        number of samples, timesteps is the number timesteps of the respective
        RNN's output, and characters is the number of known characters by the
        predictor.
    texts : np.ndarray(str)
        An iterable of strings that represents the expected lines of texts to
        be predicted.
    """
    total_cer = 0
    total_wer = 0

    for i, logit in enumerate(logits):
        pred, raw_pred = string_utils.naive_decode(logit)

        pred_str = string_utils.label2str(
            pred,
            label_encoder.encoder.inverse,
            False,
            blank_char=label_encoder.blank_char,
            blank=label_encoder.blank_idx,
        )

        total_cer += error_rates.cer(texts[i], pred_str)
        total_wer += error_rates.cer(texts[i], pred_str)

    return TranscriptResults(
        total_cer / len(texts),
        total_wer / len(texts),
        None,
    )


def eval_transcription(texts, preds):
    """Evaluates the given predicted transcriptions to expected texts.

    Parameters
    ----------
    preds : np.ndarray
        array of shape [samples], where samples is the
        number of sample lines, line_length is the length of the lines.
    texts : np.ndarray(str)
        An iterable of strings that represents the expected lines of texts to
        be predicted.
    """
    total_cer = 0
    total_wer = 0

    for i, pred in enumerate(preds):
        total_cer += error_rates.cer(texts[i], pred)
        total_wer += error_rates.cer(texts[i], pred)

    return TranscriptResults(
        total_cer / len(preds),
        total_wer / len(preds),
        None,
    )


def eval_char_confusion(texts, preds, labels=None, char_level=False):
    """Evaluates the given predicted transcriptions to expected texts by
    calculating the multiclass confusion matrix for the characters. The
    characters may be encoded.

    Parameters
    ----------
    preds : np.ndarray
        if char_level is False, then preds is an array with shape
        [samples, text_length], where samples is the number of sample text
        lines, text_length is the number of characters in the line. Otherwise,
        preds is a single dimension array with shape [samples] where samples
        corresponds to the number characters to assess.
    texts : np.ndarray(str)
        An iterable of strings that represents the expected lines of texts to
        be predicted.
    char_level : bool, optional
        True if preds and texts are to be evaluated where the first dimension
        is the number of characters, ow. preds and texts are evaluated as the
        number of lines which are looped through to evaluate the characters.
    """
    if char_level:
        assert preds.shape == texts.shape
        return ConfusionMatrix(texts, preds, labels=labels)
    else:
        # TODO expand the texts and preds to char level
        raise NotImplementedError()

    # Expand both preds and texts to be shape [characters]
    preds = np.concatenate([list(pred) for pred in preds])
    texts = np.concatenate([list(text) for text in texts])

    assert preds.shape == texts.shape

    return ConfusionMatrix(texts, preds, labels=labels)


def load_config_char_enc(config):
    if 'blank' in config['model']['crnn']['train']:
        blank = config['model']['crnn']['train']['blank']
    else:
        logging.warning(' '.join([
            'No `blank` for CTC Loss given in the config file!',
            'Default assumed is 0',
        ]))
        blank = 0

    if 'space_char' in config['model']['crnn']['train']:
        space_char = config['model']['crnn']['train']['space_char']
    else:
        logging.warning(
            'No `space_char` given in the config file! Default assumed is " "',
        )
        space_char = ' '

    if 'unknown_idx' in config['model']['crnn']['train']:
        unknown_idx = config['model']['crnn']['train']['unknown_idx']
    else:
        raise ValueError(
            'No `unknown_idx` given in the config file! No assumed default!',
        )

    char_encoder = load_char_encoder(
        config['data']['iam']['labels'],
        blank,
        space_char,
        unknown_idx,
    )

    logging.info(
        'blank: char = %s; idx = %d;',
        char_encoder.blank_char,
        char_encoder.blank_idx,
    )
    logging.info(
        'space: char = %s; idx = %d;',
        char_encoder.space_char,
        char_encoder.space_idx,
    )
    logging.info(
        'unknown: char = %s; idx = %d;',
        char_encoder.unknown_char,
        char_encoder.unknown_idx,
    )

    return char_encoder


def load_dataloader(config, char_encoder, col_chars_path=None, must_validate=True):
    if 'augmentation' in config['model']['crnn']['train']:
        train_augmentation = config['model']['crnn']['train']['augmentation']
    else:
        train_augmentation = False

    # Handle image path prefixes in config
    if 'normal_image_prefix' in config['data']['iam']:
        normal_image_prefix = config['data']['iam']['normal_image_prefix']
    else:
        normal_image_prefix = ''

    if 'antique_image_prefix' in config['data']['iam']:
        antique_image_prefix = config['data']['iam']['antique_image_prefix']
    else:
        antique_image_prefix = ''

    if 'noise_image_prefix' in config['data']['iam']:
        noise_image_prefix = config['data']['iam']['noise_image_prefix']
    else:
        noise_image_prefix = ''

    train_dataset = hw_dataset.HwDataset(
        config['data']['iam']['train'],
        char_encoder,
        img_height=config['model']['crnn']['init']['input_height'],
        root_path=config['data']['iam']['image_root_dir'],
        augmentation=train_augmentation,
        normal_image_prefix=normal_image_prefix,
        antique_image_prefix=antique_image_prefix,
        noise_image_prefix=noise_image_prefix,
        col_chars_path=col_chars_path,
    )

    try:
        test_dataset = hw_dataset.HwDataset(
            config['data']['iam']['val'],
            char_encoder,
            img_height=config['model']['crnn']['init']['input_height'],
            root_path=config['data']['iam']['image_root_dir'],
            normal_image_prefix=normal_image_prefix,
            antique_image_prefix=antique_image_prefix,
            noise_image_prefix=noise_image_prefix,
            col_chars_path=col_chars_path,
        )
    except KeyError as e:
        logging.warning("No validation set found.")

        if must_validate:
            logging.info("No validation set found, generating one")

            master = train_dataset

            logging.info("Total of " +str(len(master)) +" Training Examples")

            n = len(master)  # how many total elements you have
            n_test = int(n * .1)
            n_train = n - n_test
            idx = list(range(n))  # indices to all elements

            train_idx = idx[:n_train]
            test_idx = idx[n_train:]

            test_dataset = Subset(master, test_idx)
            train_dataset = Subset(master, train_idx)
        else:
            test_dataset = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['model']['crnn']['train']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    logging.info("Train Dataset Length: " + str(len(train_dataset)))

    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['model']['crnn']['eval']['batch_size'],
            shuffle=False,
            num_workers=1,
            collate_fn=hw_dataset.collate,
        )
        logging.info("Test Dataset Length: " + str(len(test_dataset)))

        return train_dataloader, test_dataloader

    return train_dataloader


def load_data(config, col_chars_path=None):
    char_encoder = load_config_char_enc(config)
    train_dataloader, test_dataloader = load_dataloader(
        config,
        char_encoder,
        col_chars_path=col_chars_path,
    )

    return train_dataloader, test_dataloader, char_encoder


# TODO streamline loading of files as necessary to reduce code dup & potential
# errors in typos between versions.
def load_datasplit(config, img_hieight=64):
    """Loads all datasets contained within the given data config.

    Parameters
    ----------

    """
    raise NotImplementedError('use old load deataset instead.')

    dataset = hw_dataset.HwDataset(
        config[dataset_id]['train'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config[dataset]['image_root_directory'],
        augmentation=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )
    return


def init_CRNN(config):
    """Initializes CRNN from config and loads model state if load path in
    config.
    """
    hw_crnn = CRNN(**config['model']['crnn']['init'])

    if 'load_path' in config['model']['crnn']:
        hw_crnn.load_state_dict(torch.load(
            config['model']['crnn']['load_path'],
        ))

    if torch.cuda.is_available():
        hw_crnn.cuda()
        dtype = torch.cuda.FloatTensor
        logging.info("Using GPU")
    else:
        dtype = torch.FloatTensor
        logging.info("No GPU detected")

    return hw_crnn, dtype
