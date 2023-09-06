#!/usr/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 24GB
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(resdec)
epochs=$((5000 - 1303))
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-resdec-96x54-145samples-4699epoch-52410006-8_17-1_58-EWNNP5/model-DAVE2v3-96x54-4699epoch-64batch-145Ksamples-epoch482-best012.pt"
warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-resdec-96x54-145samples-4063epoch-52615127-8_21-16_4-WZFBZN/model-DAVE2v3-96x54-4063epoch-64batch-145Ksamples-epoch204-best017.pt"
RRLdir="../data/supervised-transformation-dataset-alltransforms3/"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "96 54" --pretrained_model $pretrainedmodel --warmstart $warmstart --RRL_dir $RRLdir --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done