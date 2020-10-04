"""Script for training the MultipleEVM (MEVM) given a trained CRNN."""
from __future__ import print_function

# Python built-in packages
from copy import deepcopy
from dataclasses import dataclass
import logging

# 3rd party packages
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

def load_data(data_config):
    """Loads all datasets contained within the given data config.

    Parameters
    ----------

    """
    return

def old_load_dataset(config, dataset, RIMES=False, always_val=False):
    """Loads the datasets from the given dataset config."""
    idx_to_char, char_to_idx = character_set.load_label_set(
        config[dataset]['labels'],
    )

    train_dataset = hw_dataset.HwDataset(
        config[dataset]['training_set_path'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config[dataset]['image_root_directory'],
        augmentation=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=hw_dataset.collate,
    )

    try:
        val_dataset = hw_dataset.HwDataset(
            config[dataset]['validation_set_path'],
            char_to_idx,
            img_height=config['network']['input_height'],
            root_path=config[dataset]['image_root_directory'],
            remove_errors=True,
        )

    except KeyError as e:
        if always_val:
            print("No validation set found, generating one")
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
                shuffle=False,
                num_workers=1,
                collate_fn=hw_dataset.collate,
            )
        else:
            val_dataset = None
            val_dataloader = None

    test_dataset = hw_dataset.HwDataset(
        config[dataset]['test_set_path'],
        char_to_idx,
        img_height=config['network']['input_height'],
        root_path=config[dataset]['image_root_directory'],
        remove_errors=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=hw_dataset.collate,
    )

    return DataSet(
        idx_to_char,
        char_to_idx,
        train_dataset,
        train_dataloader, # TODO add this, be aware of this
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
