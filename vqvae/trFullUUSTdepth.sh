#!/usr/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 32GB
#SBATCH --gres=gpu:1

. /home/ms7nk/supervised-universal-transformation/.venv-sut/bin/activate
warmstart_epochs=241
epochs=$((500 - $warmstart_epochs))
transfs=(depth)
max_dataset_size="all"
basemodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
warmstart="/project/AV/supervised-universal-transformation/vqvae/results/vqvae_UUST_depth_samplesall_epochs500_52237924_fri_aug_11_19_56_51_2023/vqvae_depth_epoch241.pth"
for transf in ${transfs[@]}; do 
    echo; echo Transformation is $transf; echo max_dataset_size is full dataset
    filename=UUSTwarmstart"$warmstart_epochs"_"$transf"_samples"$max_dataset_size"_epochs"$epochs"_"$SLURM_JOB_ID"
    python main.py --dataset UUST --topo general --transf $transf --epochs $epochs --basemodel $basemodel --warmstart $warmstart -save --filename $filename
    # weights=$(ls -t results/*"$filename"*.pth | head -1)
    # echo; echo Running validation for $weights
    # python validate.py --dataset UUST --topo general --transf $transf --weights $weights --basemodel $basemodel
done