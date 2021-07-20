# Spconv-OpenPCDet
OpenPCDet with spconv package **already included** for **one-step** installation. Uses spconv & voxel CUDA ops from mmdetection3d repository.

## Purpose
I noticed that [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) had adopted parts of the [spconv](https://github.com/traveller59/spconv) repository into their cuda operations, allowing simple installation via `python setup.py develop`. In the past, I've had some trouble building spconv from the original repository on machines with scattered CUDA installations (especially when I did not have sudo privilages), so I wanted to try incorporating spconv into [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) like mmdetection3d did. 

## Changes
`pcdet/ops/spconv` and `pcdet/ops/voxel` are from the mmdetection3d repository, and `setup.py` was modified to compile the new cuda operations. To adapt calls to spconv & voxelization, `import spconv` was generally changed to `from pcdet.ops import spconv as spconv`, and voxelization usage in `pcdet/datasets/processor/data_processor.py` was slightly adjusted (only the `transform_points_to_voxels` function).

I have tested this repository on a fairly new pytorch version: pytorch 1.8, torchaudio 0.8.0, torchvision 0.9.0, cudatoolkit 11.1.1, which warranted a slight change in distributed training initialization in `common_utils.py`.

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

## Numbers
I have tested SECOND, training with 4 GPUs, 2x batch size on each GPU (to reach parity with total batch size used in OpenPCDet).
The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.

|                                             | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11 | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|
| SECOND (OpenPCDet)       |  ~1.7 hours (8 GPUs) | 78.62 | 52.98 | 67.15 |
| SECOND (Spconv-OpenPCDet) | ~3.0 hours (4 GPUs) | 78.61 | 52.03 | 64.92 |

I believe this is sufficiently close - Car is basically identical, and Pedestrian & Cyclist are known to be unstable (when evaluating last 10 epochs, Pedestrian varies 51-53 and Cyclist varies 64-67).

## Acknowledgement
This repository is basically a copy of OpenPCDet, with some elements of mmdetection3d's usage of spconv within it.
