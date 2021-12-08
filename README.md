#Semantic Point Generation


This repo consists of code for running the classification head of the paper [SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation]{https://arxiv.org/abs/2108.06709}. 

All scripts can be found in tools prefixed with spg. 

# Spconv-OpenPCDet
OpenPCDet with spconv package **already included** for **one-step** installation. Uses spconv & voxel CUDA ops from mmdetection3d repository.

## Installation
Here is how I setup for my environment:
```
conda create -n spconv-openpcdet python=3.7
conda activate spconv-openpcdet
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
python setup.py develop
```
I imagine that earlier versions of pytorch would work as well (since OpenPCDet supports 1.1, 1.3, 1.5 and mmdetection3d works for earlier versions). 

Note that mmcv, and the mm* packages do not need to be installed.

## Acknowledgement
This repository is basically a copy of OpenPCDet, with some elements of mmdetection3d's usage of spconv within it.
Thanks to [divadi]{https://github.com/Divadi/Spconv-OpenPCDet} for helping us with installation.
