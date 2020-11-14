from collections import defaultdict
import csv
import json
import logging
import os
import random

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from experiments.research.par_v1.grieggs import grid_distortion, string_utils

def collate(batch, PADDING_CONSTANT=1):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    # logging.debug(len(set([b['line_img'].shape[0] for b in batch])))
    # logging.debug(len(set([b['line_img'].shape[2] for b in batch])))
    if len(set([b['line_img'].shape[0] for b in batch])) == 0 or len(set([b['line_img'].shape[2] for b in batch])) == 0:
        return None
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    """
    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim1 = dim1 + (dim0 - (dim1 % dim0))
    dim2 = batch[0]['line_img'].shape[2]

    input_batch = np.full(
        (len(batch), dim0, dim1, dim2),
        PADDING_CONSTANT,
    ).astype(np.float32)
    #"""
    input_batch = []

    all_labels = []
    label_lengths = []
    line_ids = []

    for i in range(len(batch)):
        #b_img = batch[i]['line_img']
        input_batch.append(batch[i]['line_img'])
        #input_batch[i,:,:b_img.shape[1],:] = b_img
        line_ids.append(batch[i]['line_id'])
        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    input_batch = np.array(input_batch)
    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)

    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))
    if 'col_chars' in batch[0]:
        return {
            "line_ids": line_ids,
            "line_imgs": line_imgs,
            "labels": labels,
            "label_lengths": label_lengths,
            "gt": [b['gt'] for b in batch],
            "col_chars": [b['col_chars'] for b in batch],
        }
    return {
        "line_ids": line_ids,
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch],
    }


def load_labels_file(
    filepath,
    sep=None,
    quoting=csv.QUOTE_NONE,
    normal_prefix='',
    noise_prefix='',
    antique_prefix='',
    noise_idx=-2,
    antique_idx=-3,
):
    file_extension = filepath.rpartition('.')[-1]

    # TODO expand data given existence of True values for noise & antique
    # TODO use pandas here for simplicity...
    if file_extension in {'csv', 'tsv'}:
        if sep is None:
            sep = ',' if file_extension == 'csv' else '\t'
        with open(filepath) as f:
            reader = csv.reader(f, delimiter=sep, quoting=quoting)

            # remove headers
            next(reader)
            # NOTE ignores None or Empty string transcriptions!
            data = []
            for row in reader:
                if row[-1] is not None and row[-1] != '':
                    data.append({
                        'gt': row[-1],
                        'image_path': os.path.join(normal_prefix, row[0]),
                    })

                    if row[noise_idx] == 1:
                        data.append({
                            'gt': row[-1],
                            'image_path': os.path.join(noise_prefix, row[0]),
                        })
                    if row[antique_idx] == 1:
                        data.append({
                            'gt': row[-1],
                            'image_path': os.path.join(antique_prefix, row[0]),
                        })
    elif file_extension == 'json':
        with open(filepath, 'r') as openf:
            data = json.load(openf)
    else:
        raise ValueError(
            'HwDataset is only able to load labels from csv, tsv, or json',
        )

    return data


class HwDataset(Dataset):
    def __init__(
        self,
        json_path,
        char_encoder=None,
        img_height=32,
        img_width=None,
        root_path=".",
        augmentation=False,
        remove_errors=False,
        random_seed=None,
        normal_image_prefix='',
        antique_image_prefix='',
        noise_image_prefix='',
        col_chars_path=None,
     ):
        data = load_labels_file(
            json_path,
            normal_prefix=normal_image_prefix,
            noise_prefix=noise_image_prefix,
            antique_prefix=antique_image_prefix,
        )

        # Path prefixes
        self.root_path = root_path
        self.normal_image_prefix = normal_image_prefix
        self.antique_image_prefix = antique_image_prefix
        self.noise_image_prefix = noise_image_prefix

        self.img_height = img_height
        self.img_width = img_width
        self.char_encoder = char_encoder
        self.data = data
        self.remove_errors = remove_errors

        self.augmentation = augmentation
        self.col_chars_path = col_chars_path

        if augmentation:
            # Initialize the random state of the augmentation
            self.random_state = np.random.RandomState(random_seed)
        else:
            self.random_state = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # TODO have to figure out how to include bg_antique and noise samples!
        #   This would involve expanding such that the iterator w/ idx is blind
        #   to such an expansion of the images. So it needs done in init.
        img_path = os.path.join(
            self.root_path,
            #self.normal_image_prefix,
            item['image_path'],
        )

        img = cv2.imread(img_path)

        if self.remove_errors:
            if item['err']:
                return None
        if img is None:
            logging.warning("image is None: %s", img_path)
            return None

        if self.img_width is not None:
            img = cv2.resize(
                img,
                (self.img_width,self.img_height),
                interpolation = cv2.INTER_CUBIC,
            )
        else:
            # NOTE resize image based on given image height
            percent_x = float(self.img_height) / img.shape[0]
            img = cv2.resize(
                img,
                (0,0),
                fx=percent_x,
                fy=percent_x,
                interpolation=cv2.INTER_CUBIC,
            )

        if self.augmentation:
            img = grid_distortion.warp_image(
                img,
                h_mesh_std=5,
                w_mesh_std=10,
                random_state=self.random_state,
            )

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        if self.char_encoder is not None:
            gt = item['gt']
            gt_label = string_utils.str2label(
                gt,
                self.char_encoder.encoder,
                self.char_encoder.unknown_idx,
            )
        else:
            gt = None
            gt_label = None

        if self.col_chars_path is not None:
            if self.char_encoder is None:
                raise ValueError(
                    'col_chars_path cannot be given if char_encoder is None',
                )

            # Given bbox_dir load the image's corresponding column characters
            col_chars_full_path = os.path.join(
                self.col_chars_path,
                f'{item["image_path"].rpartition(".")[0]}.npy',
            )

            if not os.path.isfile(col_chars_full_path):
                return None

            col_chars = self.char_encoder.encode(np.load(col_chars_full_path))

            return {
                "line_id": item['image_path'],
                "line_img": img,
                "gt_label": gt_label,
                "gt": gt,
                'col_chars': col_chars,
            }

        return {
            "line_id": item['image_path'],
            "line_img": img,
            "gt_label": gt_label,
            "gt": gt,
        }
