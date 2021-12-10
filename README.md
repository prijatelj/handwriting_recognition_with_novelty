## Handwriting Recognition with Novelty

This repository contains the code for the [ICDAR 2021 paper](https://arxiv.org/abs/2105.06582), "Handwriting Recognition with Novelty" by Derek S. Prijatelj, Samuel Grieggs, Futoshi Yumoto, Eric Robertson, and Walter J. Scheirer.

### Diretory Structure

When installing out code, we recommend using a virtual environment, such as venv or conda.
The models used in experimentation are contained within `hwr_novelty`

```
hwr_novelty
├── generate
└── models
    └── losses
```

Install using `python setup.py install`

The code and scripts for the experiments are contained within `experiments`

```
experiments
├── configs
│   ├── m18_par
│   ├── paper_1
│   └── par_iam_round1
│       ├── labels
│       └── v2
│           ├── crnn
│           │   └── continue
│           └── mevm
├── crc_scripts
│   ├── mevm
│   └── paper_1
│       ├── repr
│       └── writer_id
└── research
    └── par_v1
        └── grieggs
```

- `configs` : Configuration files for running the experiments
- `crc_scripts` : Scripts used to run our code on our machines.
- `research` : The research experiment specific code

Install using `python setup_exp.py install`

### License

Our code contributions within this repository are released under the MIT License located in `LICENSE.txt`

### Citations

If you use our work, please use the following Bibtex to cite our paper:

```
@inproceedings{prijatelj_handwriting_2021,
	title = {Handwriting {Recognition} with {Novelty}},
	author = {Prijatelj, Derek S. and Grieggs, Samuel and Yumoto, Futoshi and Robertson, Eric and Scheirer, Walter J.},
	year = {2021},
	editor = {Lladós, Josep and Lopresti, Daniel and Uchida, Seiichi},
	isbn = {978-3-030-86337-1},
	doi = {10.1007/978-3-030-86337-1_33},
	booktitle = {Document {Analysis} and {Recognition} – {ICDAR} 2021},
	series = {Lecture {Notes} in {Computer} {Science}},
	publisher = {Springer International Publishing},
	pages = {494--509},
}
```
