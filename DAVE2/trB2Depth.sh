#!/usr/bin/bash
. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(depth)
epochs=$((5000 - 2446))
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
# pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
RRLdir="/p/sdbb/supervised-transformation-datasets/supervised-transformation-dataset-alltransforms3/"
# warmstart="/p/sdbb/supervised-universal-transformation/DAVE2/BASELINE2-DAVE2v3-depth-108x192-145samples-5000epoch-77376-8_6-21_40-D04PVK/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch911-best065.pt"
warmstart="/p/sdbb/supervised-universal-transformation/DAVE2/BASELINE2-DAVE2v3-depth-108x192-145samples-5000epoch-283160-8_28-17_36-HREEML/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch1453-best009.pt"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "108 192" --warmstart $warmstart --RRL_dir $RRLdir --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done