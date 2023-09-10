#!/usr/bin/bash

. ../.venv-sut/bin/activate
vqvaes="$(ls -tr ./results/*.pth)"
basemodel="../weights/model-DAVE2v3-108x192-5000epoch-64batch-145Ksamples-epoch204-best051.pt"
# vqvaes=( "/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_330993_wed_sep_6_14_01_46_2023/vqvae_fisheye_bestmodel250.pth" "/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_331029_wed_sep_6_15_43_54_2023/vqvae_fisheye_bestmodel498.pth" "/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_331046_wed_sep_6_17_39_55_2023/vqvae_fisheye_bestmodel461.pth")
a="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_332879_fri_sep_8_16_36_59_2023/vqvae_fisheye_bestmodel493.pth"
b="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_332881_fri_sep_8_16_36_59_2023/vqvae_fisheye_bestmodel495.pth"
c="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_332883_fri_sep_8_16_37_12_2023/vqvae_fisheye_bestmodel499.pth"
d="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_resinc_samples10000_epochs500_332887_fri_sep_8_17_18_47_2023/vqvae_resinc_bestmodel490.pth"
e="/p/sdbb/supervised-universal-transformation/vqvae/results/vqvae_UUST_fisheye_samples10000_epochs500_332901_fri_sep_8_18_54_12_2023/vqvae_fisheye_bestmodel495.pth"
vqvaes=($a $b $c $d $e)
for vqvae in ${vqvaes[@]}; do
    echo; echo VQVAE is $vqvae
    readarray -d _ -t strarr <<< "$vqvae"
    echo Transformation is ${strarr[2]}
    python validate.py --dataset UUST --topo general --transf ${strarr[2]} --weights $vqvae --basemodel $basemodel --id $SLURM_JOB_ID
done