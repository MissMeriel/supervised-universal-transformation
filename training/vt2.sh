#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate

python train_lenscoder.py -t /p/sdbb/supervised-transformation-validation/ -o $SLURM_JOB_ID -e 1000 -l weighted_prediction_kld -r 0.00001 -d 1024
python train_lenscoder.py -t /p/sdbb/supervised-transformation-validation/ -o $SLURM_JOB_ID -e 5000 -l weighted_prediction_kld -r 0.00001 -d 1024
python train_lenscoder.py -t /p/sdbb/supervised-transformation-validation/ -o $SLURM_JOB_ID -e 5000 -l weighted_prediction_kld -r 0.000005 -d 1024

for dir in *$SLURM_JOB_ID*/; do
  echo "Validating $dir"
  weights="$(ls $dir/*.pt)"
  python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-all -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/$dir/$weights
done