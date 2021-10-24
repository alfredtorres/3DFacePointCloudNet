# Learning Directly from Synthetic Point Clouds for "In-the-wild" 3D Face Recognition

Point clouds-based networks have achieved great attention in 3D object classification, segmentation, and indoor scene semantic parsing, but its application to 3D face recognition is still underdeveloped owing to two main reasons: lack of large-scale 3D facial data and absence of deep neural network that can directly extract discriminative face representations from point clouds. To address these two problems, a PointNet++ based network is proposed in this paper to extract face features directly from point clouds facial scans and a statistical 3D Morphable Model based 3D face synthesizing strategy is established to generate large-scale unreal facial scans to train the proposed network from scratch. A curvature-aware point sampling technique is proposed to hierarchically down-sample feature-sensitive points which are crucial to pass and aggregate discriminative facial features deeply. In addition, a novel 3D face transfer learning method is proposed to ease the domain discrepancy between synthetic and 'in-the-wild' faces.
Experimental results on two public 3D face benchmarks show that the network trained only on synthesized data can also be well generalized to 'in-the-wild' 3D face recognition. Our method achieves the state-of-the-art results by achieving an overall rank-1 identification rate of 99.46\% and 99.65\% on FRGCv2 and Bosphorus, respectively. Further, we evaluate on a self-collected dataset to demonstrate the robustness and application potential of our method.

## Data prepare
1. Download the [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) h5 file `model2017-1_face12_nomouth.h5`. Move into `/data`.  
2. Using `GenerateTrainData.m` to generate total 500,000 face scans.
3. Using [PCL](http://pointclouds.org/) to computer normal and curvature.

## Network train
### setup
Our work is based on [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch). We modify the sampling and network struture in our project.
First, we should building ext moudle  
```
python setup.py build_ext --inplace
```
then we can get a `.so` file in `/pointnet2`  
### Example training
Training examples are provided by `pointnet2/train/train_cls.py` and `pointnet2/train/train_triplet.py`. `train_cls` is used to pre-train on generated data and `train_triplet` is used to fine tune on real faces to get better results.  
```
python -m pointnet2.train.train_cls
python -m pointnet2.train.train_triplet -model_checkpoint=checkpoints/model.pth.tar  --margin=0.3 --num_triplet=30000
```
the `checkpoints/model.pth.tar` is `train_cls` result.

## Results
**Comparison of verification rates at 0.1% FAR achieved on the FRGCv2 dataset**

Method  | N vs N | N vs Non-N | N vs All 
------------- | :-----------:|:-----------: |:-----------: 
Elaiwat(2015)  | 99.4 | 94.1 | 97.1 
Guo(2016)  | 99.9 | 97.2 | 99.0 
Lei(2016) | 99.9 | 96 | 98.3 
Gilani(2017) | 99.9 | 96.6 | 98.7 
Cai(2019) | 100 | 100 | 100 
Our work | 100 | 99.1 | 99.6 

**Comparison of rank-1 identification rates achieved on the overall FRGCv2 dataset**

| Method                | Rank-1 identification rate (%) |
| --------------------- | :----------------------------: |
| Elaiwat(2015)         |              97.1              |
| Li(2015)              |              96.3              |
| Lei(2016)             |              96.3              |
| Guo(2016)             |              97.0              |
| Gilani and Mian(2018) |             97.06              |
| Cai(2019)             |              100               |
| Our work              |             99.46              |

**Comparison of the rank-1 identification rate(%) on facial expression subset of Bosphorus dataset**

Method  | Rank-1 identification rate (%) 
------------- | :-----------:
Berretti(2013)  | 95.7 
Li(2015)  | 98.8 
Lei(2016) | 98.9 
Kim(2017) | 99.2 
Gilani and Mian(2018) | 96.18 
Cai(2019) | 99.75 
Bhople(2021) | 97.55 
Our work | 99.68 

**FRGCv2 ROC curve**

![](https://github.com/alfredtorres/3DFacePointCloudNet/blob/master/img/frgc_result.png)
