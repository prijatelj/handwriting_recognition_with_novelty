#!/bin/bash
# if `bash` or `./` do not work w/ conda, use `source quick_setup.sh`
conda activate osr_torch

python3 setup.py install
python3 setup_exp.py install

conda deactivate
