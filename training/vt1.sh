#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-all/ -o $SLURM_JOB_ID -e 1000 -l orig -r 0.001 -d 64
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-all/ -o $SLURM_JOB_ID -e 5000 -l orig -r 0.0005 -d 64
python train_lenscoder.py -t /p/sdbb/supervised-transformation-dataset-all/ -o $SLURM_JOB_ID -e 5000 -l orig -r 0.00025 -d 64

for dir in *$SLURM_JOB_ID*/; do
  echo "Performing validation for trained VAE in $dir"
  weights="$(ls $dir/*.pt)"
  python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjointval -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/$dir/$weights
done