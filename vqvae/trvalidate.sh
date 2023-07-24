#!/usr/bin/bash

. /p/sdbb/BBTG/VAE/.venv-bbtg/bin/activate
vqvaes="$(ls -tr results/vqvae_RQ1v2*.pth)"
basemodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-145samples-5000epoch-5364842-7_4-17_15-XACCPQ/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
for vqvae in ${vqvaes[@]}; do
    echo; echo VQVAE is $vqvae
    readarray -d _ -t strarr <<< "$vqvae"
    echo Transformation is ${strarr[2]}
    python validate.py --dataset UUST --topo general --transf ${strarr[2]} --weights $vqvae --basemodel $basemodel --id $SLURM_JOB_ID
done