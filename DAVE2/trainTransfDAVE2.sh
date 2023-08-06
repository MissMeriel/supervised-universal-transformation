#!/usr/bin/bash
# . /p/sdbb/DAVE2-Keras/.venv-dave2/bin/activate
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(fisheye depth resdec resinc)
for transf in ${transfs[@]}; do 
    if [ $transf == "resdec" ]; then
        img_size="81 144"
        # img_size="54 96"
    elif [ $transf == "resinc" ]; then
        img_size="480 270"
    else
        img_size="108 192"
    fi
    python train_UUST_baseline.py --effect  $transf -d $img_size --pretrained_model /p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt --RRL_dir /p/sdbb/supervised-transformation-dataset-alltransforms3/ --epochs 5000 --lr 0.0001  -o $SLURM_JOB_ID 
done