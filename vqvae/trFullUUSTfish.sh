#!/usr/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 512GB
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
epochs=500
transfs=(fisheye)
max_dataset_size="all"
basemodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
warmstart="/home/ms7nk/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samplesall_epochs500_52103396_tue_aug_8_13_12_05_2023.pth"
for transf in ${transfs[@]}; do 
    echo; echo Transformation is $transf; echo max_dataset_size is full dataset
    filename=UUST_"$transf"_samples"$max_dataset_size"_epochs"$epochs"_"$SLURM_JOB_ID"
    python main.py --dataset UUST --topo general --transf $transf --epochs $epochs --basemodel $basemodel -save --filename $filename
    # weights=$(ls -t results/*"$filename"*.pth | head -1)
    # echo; echo Running validation for $weights
    # python validate.py --dataset UUST --topo general --transf $transf --weights $weights --basemodel $basemodel
done