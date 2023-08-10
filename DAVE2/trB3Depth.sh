#!/usr/bin/bash
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(depth)
epochs=2085
baseweights="/p/sdbb/supervised-universal-transformation/DAVE2/BASELINE3-DAVE2v3-depth-108x192-50samples-5000epoch-26322-8_4-15_20-FKS06T/model-DAVE2v3-108x192-5000epoch-64batch-50Ksamples-epoch2915.pt"
for transf in ${transfs[@]}; do
    if [ $transf == "resdec" ]; then
        img_size="54 96"
    elif [ $transf == "resinc" ]; then
        img_size="270 480"
    else
        img_size="108 192"
    fi
    echo TRAINING $transf transformation with img size $img_size
    python train_baseline3.py --effect  $transf -d "$img_size"  --pretrained_model $baseweights  --RRL_dir /p/sdbb/supervised-transformation-dataset-alltransforms31FULL-V/ --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done