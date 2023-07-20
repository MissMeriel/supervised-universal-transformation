#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-all/ -v /p/sdbb/supervised-transformation-validation -o $SLURM_JOB_ID -e 1000 -l weighted_prediction_kld -r 0.0001
# python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset/ -v /p/sdbb/supervised-transformation-validation -o $SLURM_JOB_ID -e 1000 -l prediction_kld -r 0.0001
dir="$(ls -a *$SLURM_JOB_ID*/*.pt)"
python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjointval -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/$dir