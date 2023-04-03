#!/usr/bin/bash
 . ../.venv-sut/bin/activate
# python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjoint -o $SLURM_JOB_ID -w /p/sdbb/samples_pretrain-Lenscoder-origloss-4930193-3_31-21_55-OSZ36G/Lenscoder_featureloss_38k-512lat_1000epochs_32batch_Falserob.pt
python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjoint -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/samples_Lenscoder-prediction_kldloss-dataindist-4930078-3_31-11_22-BFSM9G/Lenscoder_featureloss_38k-512lat_1000epochs_32batch_Falserob.pt

