#!/usr/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH -A lesslab
#SBATCH -p gpu
#SBATCH --mem 500GB
#SBATCH --gres=gpu:1

. ../.venv-sut/bin/activate
# SLURM_JOB_ID="000"
transfs=(resinc)
epochs=$((5000 - 397))
# pretrainedmodel="/p/sdbb/DAVE2-Keras/DAVE2v3-108x192-82samples-5000epoch-5116933-4_12-23_11-2D8U63/model-DAVE2v3-randomblurnoise-108x192-lr1e4-5000epoch-64batch-lossMSE-82Ksamples-epoch797.pt"
pretrainedmodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-resinc-270x480-145samples-4976epoch-52410030-8_17-3_13-NP5NIH/model-DAVE2v3-270x480-4976epoch-64batch-145Ksamples-epoch299-best045.pt"
warmstart="/project/AV/supervised-universal-transformation/DAVE2/results_baseline2/BASELINE2-DAVE2v3-resinc-270x480-145samples-4636epoch-52615191-8_21-16_9-VSVZXO/model-DAVE2v3-270x480-4636epoch-64batch-145Ksamples-epoch025-best008.pt"
RRLdir="../data/supervised-transformation-dataset-alltransforms3/"
for transf in ${transfs[@]}; do 
    python train_baseline2.py --effect  $transf -d "270 480" --pretrained_model $pretrainedmodel --warmstart $warmstart --RRL_dir $RRLdir --epochs $epochs --lr 0.0001  -o $SLURM_JOB_ID 
done