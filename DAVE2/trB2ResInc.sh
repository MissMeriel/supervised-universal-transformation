#!/usr/bin/bash
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(resinc)
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
RRLdir="../data/supervised-transformation-dataset-alltransforms3/"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "270 480" --pretrained_model $pretrainedmodel --RRL_dir $RRLdir --epochs 5000 --lr 0.0001  -o $SLURM_JOB_ID 
done