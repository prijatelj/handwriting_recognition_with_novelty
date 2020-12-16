"""Data handler and analysis of the IAM Handwriting Database."""
from copy import deepcopy
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from exputils.data import NumpyJSONEncoder

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

class IAMHandwriting(object):
    def __init__(self, filepath):
        if isinstance(filepath, np.ndarray):
            self.files = filepath
        else:
            with open(filepath, 'r') as openf:
                labels_dict = json.load(openf)

            self.files = np.array([
                os.path.splitext(
                    os.path.basename(x['image_path'])
                )[0] for x in labels_dict
            ])

        self.df = pd.DataFrame(
            [f.split('-') for f in self.files],
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
            self.files[strat_idx],
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
                    self.files[st_tr],
                    self.files[[
                        val in label_set_sep[sep_kf[0]] for val in self.df[col]
                    ]]
                ]),
                np.concatenate([
                    self.files[st_te],
                    self.files[[
                        val in label_set_sep[sep_kf[1]] for val in self.df[col]
                    ]]
                ]),
            ))

        return folds

if __name__ == '__main__':
    # IAM
    iamh_tr = IAMHandwriting(train_path)
    iamh_val = IAMHandwriting(val_path)
    iamh_te = IAMHandwriting(test_path)
    ok = iamh_tr + iamh_val + iamh_te
    folds = ok.kfold(5, seed=0)

    for i, (train, test) in enumerate(folds):
        with open(f'../tmp/paper/iam_splits/iam_split_{i}.txt', 'w') as f:
            train, val = IAMHandwriting(train).kfold(5, seed=0)[0]
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
