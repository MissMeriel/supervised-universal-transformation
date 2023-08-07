#!/usr/bin/bash
. /p/sdbb/BBTG/VAE/.venv-bbtg/bin/activate
epochs=500
transfs=(resdec fisheye resinc depth)
max_dataset_size="all"
basemodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-145samples-5000epoch-5364842-7_4-17_15-XACCPQ/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
for transf in ${transfs[@]}; do 
    echo; echo Transformation is $transf; echo max_dataset_size is full dataset
    filename=RQ1v2_"$transf"_samples"$max_dataset_size"_epochs"$epochs"_"$SLURM_JOB_ID"
    python main.py --dataset UUST --topo general --transf $transf --epochs $epochs --basemodel $basemodel -save --filename $filename
    weights=$(ls -t results/*"$filename"*.pth | head -1)
    echo; echo Running validation for $weights
    python validate.py --dataset UUST --topo general --transf $transf --weights $weights --basemodel $basemodel
done