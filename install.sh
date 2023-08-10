#!/usr/bin/bash

python3.8 -m venv .venv-sut
. .venv-sut/bin/activate
# pip3 install tensorboard kornia numpy black mypy matplotlib scipy scikit-image pandas opencv-python markupsafe==2.0.1
# pip3 install torch==2.0.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
pip install -r portalreqs.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install --upgrade torch torchvision
pip install blurgenerator discorpy scikit-learn