#!/bin/bash
CONDA_BASE=$(conda info --base); source $CONDA_BASE/etc/profile.d/conda.sh
conda create -n hydra python=3.8 -y
#source activate hydra
conda activate hydra
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
pip install tensorflow
pip install transformers
pip install Cython pycocotools
echo "CONDA_BASE=$(conda info --base); source $CONDA_BASE/etc/profile.d/conda.sh"
echo "conda activate hydra"