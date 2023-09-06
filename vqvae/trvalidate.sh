#!/usr/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB

. ../.venv-sut/bin/activate
vqvaes="$(ls -tr ./results/*.pth)"
basemodel="./model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
for vqvae in ${vqvaes[@]}; do
    echo; echo VQVAE is $vqvae
    readarray -d _ -t strarr <<< "$vqvae"
    echo Transformation is ${strarr[2]}
    python validate.py --dataset UUST --topo general --transf ${strarr[2]} --weights $vqvae --basemodel $basemodel --id $SLURM_JOB_ID
done