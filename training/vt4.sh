#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-indistribution/ -v /p/sdbb/supervised-transformation-validation -o $SLURM_JOB_ID -e 1000 -l prediction_only -r 0.0001
dir=$(ls -d *$SLURM_JOB_ID*/)
echo
echo "Validating $dir"
python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjointval -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/$dir