#!/usr/bin/bash

. /home/ms7nk/supervised-universal-transformation/.venv-sut/bin/activate
warmstart_epochs=241
epochs=$((500))
transfs=(depth)
max_dataset_size="all"
pred_weight=1.0
basemodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# warmstart="/project/AV/supervised-universal-transformation/vqvae/results/vqvae_UUST_depth_samplesall_epochs500_52237924_fri_aug_11_19_56_51_2023/vqvae_depth_epoch241.pth"
for transf in ${transfs[@]}; do 
    echo; echo Transformation is $transf; echo max_dataset_size is full dataset
    filename=UUST_predweight"$pred_weight"_"$transf"_samples"$max_dataset_size"_epochs"$epochs"_"$SLURM_JOB_ID"
    python main.py --dataset UUST --topo general --transf $transf  --pred_weight $pred_weight --epochs $epochs --basemodel $basemodel -save --filename $filename
    # weights=$(ls -t results/*"$filename"*.pth | head -1)
    # echo; echo Running validation for $weights
    # python validate.py --dataset UUST --topo general --transf $transf --weights $weights --basemodel $basemodel
done