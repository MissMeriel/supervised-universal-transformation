#!/usr/bin/bash
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-indistribution/ -v /p/sdbb/supervised-transformation-validation -o $SLURM_PROCID -e 1000 -l orig -r 0.0001