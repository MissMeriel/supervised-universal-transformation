#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-indistribution/ -v /p/sdbb/supervised-transformation-validation -o $SLURM_JOB_ID -e 1000 -l prediction_kld -r 0.0001