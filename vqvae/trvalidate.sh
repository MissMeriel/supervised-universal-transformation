#!/usr/bin/bash


. /p/sdbb/BBTG/VAE/.venv-bbtg/bin/activate
vqvaes="$(ls -tr ./results/*.pth)"
basemodel="./model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
vqvae="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_RQ1v2_resdec_samples10000_epochs500_5444459_fri_jul_21_19_05_57_2023.pth"
# for vqvae in ${vqvaes[@]}; do
echo; echo VQVAE is $vqvae
readarray -d _ -t strarr <<< "$vqvae"
echo Transformation is ${strarr[2]}
python validate.py --dataset UUST --topo general --transf ${strarr[2]} --weights $vqvae --basemodel $basemodel --id $SLURM_JOB_ID
# done