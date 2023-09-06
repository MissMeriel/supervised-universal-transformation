#!/usr/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 100GB
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(depth)
epochs=$((5000 - 1765))
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-depth-108x192-145samples-4985epoch-52409890-8_17-1_56-SW02GX/model-DAVE2v3-108x192-4985epoch-64batch-145Ksamples-epoch1330-best042.pt"
warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-depth-108x192-145samples-3635epoch-52615087-8_21-16_1-1OKIXE/model-DAVE2v3-108x192-3635epoch-64batch-145Ksamples-epoch316-best009.pt"
RRLdir="../data/supervised-transformation-dataset-alltransforms3/"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "108 192" --pretrained_model $pretrainedmodel --warmstart $warmstart --RRL_dir $RRLdir --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done