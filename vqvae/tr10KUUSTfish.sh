#!/usr/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
epochs=500
max_dataset_sizes=(10000)
transfs=(fisheye)
basemodel="./weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
for transf in ${transfs[@]}; do 
    for max_dataset_size in ${max_dataset_sizes[@]}; do
        echo; echo Transformation is $transf; echo max_dataset_size is $max_dataset_size
        filename=UUST_"$transf"_samples"$max_dataset_size"_epochs"$epochs"_"$SLURM_JOB_ID"
        python main.py --dataset UUST --topo general --transf $transf --epochs $epochs --basemodel $basemodel --max_dataset_size $max_dataset_size -save --filename $filename
        weights=$(ls -t results/*"$filename"*.pth | head -1)
        echo; echo Running validation for $weights
        python validate.py --dataset UUST --topo general --transf $transf --weights $weights --basemodel $basemodel
    done
done
