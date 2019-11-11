from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import sys
sys.path.append('/home/zzy/file/experiment/PointNet/Pointnet2_PyTorch-master_Curvature')
from torchvision import transforms
import torch

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
import time
from random import shuffle
from pointnet2.data import data_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
       
def readbcn(file):
    npoints = os.path.getsize(file) // 4
    with open(file,'rb') as f:
        raw_data = struct.unpack('f'*npoints,f.read(npoints*4))
        data = np.asarray(raw_data,dtype=np.float32)       
#    data = data.reshape(len(data)//6, 6)
    data = data.reshape(7, len(data)//7)
    # translate the nose tip to [0,0,0]
#    data = (data[:,0:2] - data[8157,0:2]) / 100
    return torch.from_numpy(data.T)

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(dir, class_to_idx, extensions=None):
    images = []
    dir = os.path.expanduser(dir)
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images
    

class GPMMNormalCurvDataset(data.Dataset):
    def __init__(self, root, class_nums = 500, transforms=None, train = True, extensions='bcnc'):
        self.root = root
        self.class_nums = class_nums
        self.transforms = transforms
        self.train = train
        
        classes, class_to_idx = self._find_classes(self.root, self.class_nums)
        samples = make_dataset(self.root, class_to_idx, extensions)
        shuffle(samples)
        if train:
            samples = samples[:int(len(samples) * 0.8)]
        else:
            samples = samples[int(len(samples) * 0.8):]
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of:" + self.root + "\n"
                               "Supported extensions areL" + ",".join(extensions)))
        self.extensions = extensions
        self.classes = classes
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.data_length = len(self.samples)
#        print(samples)
        ## load GPMM model
#        f = h5py.File('/home/zzy/file/experiment/3DMM-Generate/code/model2017-1_face12_nomouth.h5','r') 
#        self.shape = f['shape']['model']
#        self.shape_mean = torch.from_numpy(self.shape['mean'][:])
#        self.shape_pcaBasis = torch.from_numpy(self.shape['pcaBasis'][:])
#        self.shape_pcaVariance = torch.from_numpy(self.shape['pcaVariance'][:])
#        
#        self.expression = f['expression']['model']
#        self.expression_mean = torch.from_numpy(self.expression['mean'][:])
#        self.expression_pcaBasis = torch.from_numpy(self.expression['pcaBasis'][:])
#        self.expression_pcaVariance = torch.from_numpy(self.expression['pcaVariance'][:])

    def _find_classes(self, dir, class_nums):
        if sys.version_info >= (3,5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes = classes[:class_nums]
        class_to_idx = {classes[i] : i for i in range(len(classes))}   
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = readbcn(path)
        #resample
        choice = np.random.choice(len(sample), len(sample), replace=False)       
        sample = sample[choice, :]#choice
        
        sample[:,0:3] = (sample[:,0:3])/(100)
        sample[:,6] = torch.pow(sample[:,6],0.1)
#        sample[:,6] = (sample[:,6] - min(sample[:,6]))/(max(sample[:,6]) - min(sample[:,6]))
        if self.transforms is not None:
            point_set = self.transforms(sample)
        else:
            point_set = sample
        return point_set, target

    def __len__(self):
        return self.data_length
    

if __name__ == '__main__':
    print('test')
    transforms = transforms.Compose(
        [
#            d_utils.PointcloudToTensor(),
            data_utils.PointcloudRotate(axis=np.array([1, 1, 1])),
#            d_utils.PointcloudScale(),
#            d_utils.PointcloudTranslate(),
            data_utils.PointcloudJitter(std=0.002),
        ]
    )
    train_dataset = GPMMNormalCurvDataset(root = '/home/zzy/m2_500/GPMM10000_bcnc',
                    class_nums = 10000,
                    transforms=transforms,
                    train=True,
                    extensions='bcnc')
    print(train_dataset[0][1])
    print(train_dataset.data_length)
    test_dataset = GPMMNormalCurvDataset(root = '/home/zzy/m2_500/GPMM10000_bcnc',
                    class_nums = 1500,
                    transforms=None,
                    train=False,
                    extensions='bcnc')
    for i in range(len(train_dataset)):
        data = train_dataset[i][0]
        try:
            c_max = torch.max(data[:,6])
        except:
            print(train_dataset[i][2])
        
        if c_max==0:
            print(i)
        if i%1000==0:
            print(i)
#    print(test_dataset[0][0])
#    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
#            num_workers=4,
#            pin_memory=True,)
# =============================================================================
#     for epoch in range(5):        
#         
#         start = time.time()
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             input      = input.cuda()
#             target     = target.cuda()
#             print('batch:',i)
#             print('batch time:',time.time() - end)
#             end = time.time()
#         print('epoch[{}] train time: {}'.format(epoch ,time.time() - start) )
# =============================================================================
# =============================================================================
#     face_pt = train_dataset[0][0].numpy()
#     fig1 = plt.figure(1)
#     ax1 = Axes3D(fig1)
#     # ax.scatter(face_pt[:,0],face_pt[:,1],face_pt[:,2],c='b',marker='.')
#     # the 1st SAmodule sampling
#     face_fps = face_pt.squeeze()
#     print('After SSG1 face shape is :', face_fps.shape)
#     ax1.scatter(face_fps[:,0],face_fps[:,1],face_fps[:,2],c=face_fps[:,5],marker='.')
#     c = face_fps[:,6]
#     print(face_fps[:,6])
# =============================================================================

