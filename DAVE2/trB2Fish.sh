#!/usr/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 150GB
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(fisheye)
epochs=$((5000 - 2935))
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-fisheye-108x192-145samples-3809epoch-52410120-8_17-3_13-DB8VN1/model-DAVE2v3-108x192-3809epoch-64batch-145Ksamples-epoch443-best015.pt"
# warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-fisheye-108x192-145samples-3283epoch-52614972-8_21-15_53-9Y7LN5/model-DAVE2v3-108x192-3283epoch-64batch-145Ksamples-epoch961-best017.pt"
warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-fisheye-108x192-145samples-3283epoch-52766600-8_28-15_36-9925MC/model-DAVE2v3-108x192-3283epoch-64batch-145Ksamples-epoch003-best007.pt"
RRLdir="../data/supervised-transformation-dataset-alltransforms3/"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "108 192" --pretrained_model $pretrainedmodel --warmstart $warmstart --RRL_dir $RRLdir --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done