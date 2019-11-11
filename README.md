# 3DFacePointCloudNet

Point clouds-based Networks have achieved great attention in 3D object classification, segmentation and indoor scene semantic parsing. In terms of face recognition, 3D face recognition method which directly consume point clouds as input is still under study. Two main factors account for this: One is how to get discriminative face representations from 3D point clouds using deep network; the other is the lack of large 3D training dataset. To address these problems, a data-free 3D face recognition method is proposed only using synthesized unreal data from statistical 3D Morphable Model to train a deep point cloud network. To ease the inconsistent distribution between model data and real faces, different point sampling methods are used in train and test phase. In this paper, we propose a curvature-aware point sampling(CPS) strategy replacing the original furthest point sampling(FPS) to hierarchically down-sample feature-sensitive points which are crucial to pass and aggregate features deeply. A PointNet++ like Network is used to extract face features directly from point clouds. The experimental results show that the network trained on generated data generalizes well for real 3D faces. Fine tuning on a small part of FRGCv2.0 and Bosphorus, which include real faces in different poses and expressions, further improves recognition accuracy.

## data prepare
1. Download the [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) h5 file `model2017-1_face12_nomouth.h5`. Move into `/data`.  
2. Using `GenerateTrainData.m` to generate total 500,000 face scans.
3. Using [PCL](http://pointclouds.org/) to computer normal and curvature.

## network train
### setup
Our work is based on [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch). We modify the sampling and network struture in our project.
First, we should building ext moudle  
```
python setup.py build_ext --inplace
```
then we can get a `.so` file in `/pointnet2`  
### example training
Training examples are provided by `pointnet2/train/train_cls.py` and `pointnet2/train/train_triplet.py`. `train_cls` is used to pre-train on generated data and `train_triplet` is used to fine tune on real faces to get better results.  
```
python -m pointnet2.train.train_cls
python -m pointnet2.train.train_triplet -model_checkpoint=checkpoints/model.pth.tar  --margin=0.3 --num_triplet=30000
```
the `checkpoints/model.pth.tar` is `train_cls` result.

## Results
**FRGCv2**

Method  | Top1 rate  (%)
------------- | -------------
w/o fine tune  | 92.74
with fine tune  | 98.73

**Bosphorus**

Method  | Top1 rate  (%)
------------- | -------------
w/o fine tune  | 93.38
with fine tune  | 97.50

**FRGCv2 ROC curve**

![](https://github.com/alfredtorres/3DFacePointCloudNet/blob/master/img/frgc_result.png)
