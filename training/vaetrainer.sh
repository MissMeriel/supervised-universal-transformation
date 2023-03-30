#!/usr/bin/bash
python train_lenscoder.py -p ../supervised-transformation-dataset-indistribution/ -o $SLURM_PROCID
