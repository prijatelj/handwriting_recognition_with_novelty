"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
from __future__ import print_function

# Python built-in packages
from copy import deepcopy
from dataclasses import dataclass
import logging

# 3rd party packages
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Internal package modules
from hwr_novelty.models.crnn import CRNN

from experiments.research.par_v1.grieggs import character_set, hw_dataset


@dataclass
class DataSet:
    """Contains everything for the dataset handling in one place."""
    idx_to_char: dict
    char_to_idx: dict
    train_dataset: hw_dataset.HwDataset
    train_dataloader: DataLoader
    val_dataset: hw_dataset.HwDataset
    val_dataloader: DataLoader
    test_dataset: hw_dataset.HwDataset
    test_dataloader: DataLoader


@dataclass
class TranscriptResults:
    """Contains everything for the dataset handling in one place."""
    char_errors: int
    word_errors: int
    char_confusion_mat: np.ndarray


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

def old_load_dataset(config, dataset, shuffle=False, always_val=False):
    """Loads the datasets from the given dataset config."""
    idx_to_char, char_to_idx = character_set.load_label_set(
        config[dataset]['labels'],
    )

    data_exists = False

    if 'train' in config['data'][dataset]:
        train_dataset = hw_dataset.HwDataset(
            config['data'][dataset]['training_set_path'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config['data'][dataset]['image_root_directory'],
            augmentation=False,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=1,
            collate_fn=hw_dataset.collate,
        )
        data_exists = True
    else:
        train_dataset = None
        train_dataloader = None

    if 'val' in config['data'][dataset]:
        val_dataset = hw_dataset.HwDataset(
            config['data'][dataset]['val'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config['data'][dataset]['image_root_directory'],
            #augmentation=False,
            remove_errors=True,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=1,
            collate_fn=hw_dataset.collate,
        )

        data_exists = True
    elif always_val:
        print("No validation set found, generating one")
        if data_exists:
            master = train_dataset

            print("Total of " +str(len(master)) +" Training Examples")
            n = len(master)  # how many total elements you have
            n_test = int(n * .1)
            n_train = n - n_test

            idx = list(range(n))  # indices to all elements
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            val_dataset = Subset(master, test_idx)
            train_dataset = Subset(master, train_idx)

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=shuffle,
                num_workers=1,
                collate_fn=hw_dataset.collate,
            )
        else:
            raise KeyError('No training set exists, but `always_val` is True')
    else:
        val_dataset = None
        val_dataloader = None

    if 'test' in config['data'][dataset]:
        test_dataset = hw_dataset.HwDataset(
            config['data'][dataset]['test'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config['data'][dataset]['image_root_directory'],
            remove_errors=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=0,
            collate_fn=hw_dataset.collate,
        )
        data_exists = True
    else:
        test_dataset = None
        test_dataloader = None

    if not data_exists:
        raise KeyError('No train, val, or test set given in config!')

    return DataSet(
        idx_to_char,
        char_to_idx,
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        test_dataset,
        test_dataloader,
    )


def load_prepare_data(config):
    # Load Data
    datasets = {}
    if 'iam' in config:
        iam_dset = old_load_dataset(config, 'iam')
    else:
        raise NotImplementedError('Need to have IAM dataset in config.')

    if 'rimes' in config:
        datasets['rimes'] = old_load_dataset(config, 'rimes')
    if 'manuscript' in config:
        datasets['manuscript'] = old_load_dataset(config, 'manuscript')

    # Combine the char to idx and idx to chars iam_dset = datasets.pop('iam')
    all_dsets = deepcopy(iam_dset)
    inc = len(all_dsets.char_to_idx) + 1
    for dset in datasets:
        # Remove all char keys that already exist in char to idx
        for key in (
            all_dsets.char_to_idx.keys() & datasets[dset].char_to_idx.keys()
        ):
            datasets[dset].char_to_idx.pop(key, None)

        # Recreate indices based on remaining chars to indices
        for key in datasets[dset].char_to_idx:
            datasets[dset].char_to_idx[key] = inc
            inc += 1

        all_dsets.char_to_idx.update(datasets[dset].char_to_idx)

    datasets['iam'] = iam_dset

    # Create idx_to_char from char_to_idx
    all_dsets.idx_to_char = {v: k for k, v in all_dsets.char_to_idx.items()}

    # NOTE possiblity of these needing to be from iam_dset
    #char_to_idx = all_dsets.char_to_idx
    #idx_to_char = all_dsets.idx_to_char

    # ??? combine train, val, and test to form knowns and unknowns
    # datasets, dataloaders
    # The datasets could be as simple as a map to the filepaths and just stack
    # the 3 datasets lists together

    return iam_dset, all_dsets


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
