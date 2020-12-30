"""Data handler and analysis of the IAM Handwriting Database."""
from copy import deepcopy
from collections import namedtuple
from dataclasses import dataclass
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from exputils.io import NumpyJSONEncoder

# TODO refactor and reorganize this.

@dataclass
class Summary:
    """Class for storing data summary info."""
    samples: int
    samples_per_doc: pd.Series
    samples_per_writer: pd.Series
    #writer_per_doc: BidirDict
    #lines_per_d0c:

    #def doc_per_writer(self) -> dict:
    #    return self.writer_per_doc.inverse()

@dataclass
class HWRItem:
    """Data class for the item returned by the HWR class get item."""
    image: np.ndarray
    text: str
    path: str
    writer: str
    represent: str = None

class HWRHandwriting(object):
    def __init__(self, filepath, key=None):
        if isinstance(filepath, np.ndarray):
            self.ids = filepath
            self.texts = None
            self.files = None
        else:
            with open(filepath, 'r') as openf:
                labels_dict = json.load(openf)
                if key is not None and isinstance(labels_dict, dict):
                    labels_dict = labels_dict[key]

            files = []
            texts = []
            ids = []
            for x in labels_dict:
                path = x['image_path']
                files.append(path)
                ids.append(os.path.splitext(os.path.basename(path))[0])
                texts.append(x['gt'])
            self.ids = np.array(ids)
            self.files = np.array(files)
            self.texts = np.array(texts)

        if 'imes' in filepath.lower():
            self.df = None
        else:
            self.df = pd.DataFrame(
                [f.split('-') for f in self.ids],
                columns=['doc', 'writer', 'line'],
            )

    def __add__(self, other):
        if type(self) != type(other):
            raise TypeError(' '.join([
                'unsupported operand type(s) for +:',
                f'{type(self)} and {type(other)}'
            ]))

        df = deepcopy(self)
        df.df = self.df.append(other.df)
        df.files = np.concatenate((self.files, other.files))
        df.texts = np.concatenate((self.texts, other.texts))
        df.ids = np.concatenate((self.ids, other.ids))
        return df

    def doc_summary(self):
        # samples per doc
        # writer per doc and number of lines per writer
        # lines per doc (sum of the lines per writer per doc)
        # number of writers per doc
        return

    def summary(self):
        return
        #return Summary(
        #    len(self.df),
        #    len(set(self.df['doc'])),
        #    len(set(self.df['writer'])),
        #)

    def kfold(self, kfold, col='writer', shuffle=True, seed=None):
        """stratified fold that respects same doc in same split."""
        label_set = np.unique(self.df[col])

        # Split in half the unique column values
        strat, sep = next(KFold(2, shuffle, seed).split(label_set))

        label_set_strat = set(label_set[strat])
        label_set_sep = label_set[sep]

        # 1st half is k fold stratified
        strat_idx= [val in label_set_strat for val in self.df[col]]
        strat_kfold = StratifiedKFold(kfold, shuffle, seed).split(
            self.ids[strat_idx],
            self.df[col][strat_idx],
        )

        # 2nd half is split into k folds of unique writers in each fold.
        # Split the writers into separate folds and then index based on them.
        sep_kfold = KFold(kfold, shuffle, seed).split(label_set_sep)

        folds = []
        for i, sep_kf in enumerate(sep_kfold):
            st_tr, st_te = next(strat_kfold)
            folds.append((
                np.concatenate([
                    self.ids[st_tr],
                    self.ids[[
                        val in label_set_sep[sep_kf[0]] for val in self.df[col]
                    ]]
                ]),
                np.concatenate([
                    self.ids[st_te],
                    self.ids[[
                        val in label_set_sep[sep_kf[1]] for val in self.df[col]
                    ]]
                ]),
            ))

        return folds


class HWR(object):
    def __init__(
        self,
        filepath,
        datasplit,
        root_path,
        img_height=32,
        augmentation=False,
    ):
        iam_hw = HWRHandwriting(filepath, datasplit)
        self.df = iam_hw.df
        self.files = iam_hw.files
        self.texts = iam_hw.texts
        self.ids = iam_hw.ids

        self.root_path = root_path
        self.img_height = img_height
        self.augmentation = augmentation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_path, self.files[idx])
        image = cv2.imread(img_path)

        if image is None:
            raise IOError(f'`image` is None. image path: {img_path}')

        # Resize image based on given image height
        percent_x = float(self.img_height) / image.shape[0]
        image = cv2.resize(
            image,
            (0,0),
            fx=percent_x,
            fy=percent_x,
            interpolation=cv2.INTER_CUBIC,
        )

        if self.augmentation:
            image = grid_distortion.warp_image(
                image,
                h_mesh_std=5,
                w_mesh_std=10,
                random_state=self.random_state,
            )

        image = image.astype(np.float32) / 128.0 - 1.0

        return HWRItem(
            image,
            self.texts[idx],
            img_path,
            None if self.df is None else self.df['writer'][idx],
        )

    @property
    def values(self):
        return np.array(list(self))


if __name__ == '__main__':
    # HWR
    iamh_tr = HWRHandwriting(train_path)
    iamh_val = HWRHandwriting(val_path)
    iamh_te = HWRHandwriting(test_path)
    ok = iamh_tr + iamh_val + iamh_te
    folds = ok.kfold(5, seed=0)

    for i, (train, test) in enumerate(folds):
        with open(f'../tmp/paper/iam_splits/iam_split_{i}.txt', 'w') as f:
            train, val = HWRHandwriting(train).kfold(5, seed=0)[0]
            json.dump(
                {
                    'train': train,
                    'val' : val,
                    'test' : test,
                },
                f,
                cls=NumpyJSONEncoder,
                indent=4,
            )

    # RIMES
    for i, (train, test) in enumerate(kf_gen):
        with open(f'../tmp/paper/rimes_splits/rimes_split_{i}.txt', 'w') as f:
            train2, val = next(KFold(5, shuffle=True, random_state=0).split(rimes[train]))
            json.dump(
                {
                    'train': rimes[train][train2],
                    'val' : rimes[train][val],
                    'test' : rimes[test],
                },
                f,
                cls=NumpyJSONEncoder,
                indent=4,
            )
