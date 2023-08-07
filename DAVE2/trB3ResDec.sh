#!/usr/bin/bash
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(resdec)
for transf in ${transfs[@]}; do
    if [ $transf == "resdec" ]; then
        # img_size="144 81"
        img_size="54 96"
    elif [ $transf == "resinc" ]; then
        img_size="270 480"
    else
        img_size="108 192"
    fi
    echo TRAINING $transf transformation with img size $img_size
    python train_UUST_baseline3.py --effect  $transf -d "$img_size" --RRL_dir /p/sdbb/supervised-transformation-dataset-alltransforms31FULL-V/ --epochs 5000 --lr 0.0001  -o $SLURM_JOB_ID 
done