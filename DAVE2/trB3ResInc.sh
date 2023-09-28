#!/usr/bin/bash
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(resinc)
weights="/p/sdbb/supervised-universal-transformation/DAVE2/BASELINE3-DAVE2v3-resinc-480x270-50samples-5000epoch-26422-8_4-15_27-XHNFE1/model-DAVE2v3-480x270-5000epoch-64batch-50Ksamples-epoch3116-best118.pt"
epochs=$(( 5000 - 3182 ))
for transf in ${transfs[@]}; do
    if [ $transf == "resdec" ]; then
        # img_size="81 144"
        img_size="54 96"
    elif [ $transf == "resinc" ]; then
        img_size="270 480"
    else
        img_size="108 192"
    fi
    echo TRAINING $transf transformation with img size $img_size
    python train_baseline3.py --effect  $transf -d "$img_size" --RRL_dir /p/sdbb/supervised-transformation-dataset-alltransforms31FULL-V/ --pretrained_model $weights --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done