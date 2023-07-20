#!/usr/bin/bash
. /p/sdbb/supervised-universal-transformation/.venv-sut/bin/activate
python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjointval -o $SLURM_JOB_ID -w samples_all-Lenscoder-empiricalTEST_weighted_prediction_kldloss-5110632-4_10-13_44-IMEUBO/Lenscoder_40k-512lat_1000epochs_32batch_Falserob.pt