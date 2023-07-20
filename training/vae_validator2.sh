#!/usr/bin/bash
 . ../.venv-sut/bin/activate
python validate_lenscoder.py -v /p/sdbb/supervised-transformation-validation -d /p/sdbb/supervised-transformation-dataset-disjointval -o $SLURM_JOB_ID -w /p/sdbb/supervised-universal-transformation/training/samples_all-Lenscoder-latent_kldloss-4930706-4_5-15_58-SWZLUX/Lenscoder_40k-512lat_1000epochs_32batch_Falserob.pt 