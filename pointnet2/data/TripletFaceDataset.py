from __future__ import print_function

import torch.utils.data as data
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import struct

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def readbcn(file):
    npoints = os.path.getsize(file) // 4
    with open(file,'rb') as f:
        raw_data = struct.unpack('f'*npoints,f.read(npoints*4))
        data = np.asarray(raw_data,dtype=np.float32)       
    data = data.reshape(7, len(data)//7)
    return torch.from_numpy(data.T)

class TripletFaceDataset(data.Dataset):
    def __init__(self, root, n_triplets, transforms=None, n_points=20000, class_nums = 500, extensions='bcnc'):
        self.root = root
        self.class_nums = class_nums
        self.transforms = transforms
        self.n_triplets = n_triplets
        self.npoints = n_points
        
        classes, class_to_idx = self._find_classes(self.root, self.class_nums)
        indices = self.create_indices(root, class_to_idx, extensions)

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = self.generate_triplets(indices, self.n_triplets,len(classes))
#        training_triplets = self.training_triplets
        
    def _find_classes(self, dir, class_nums):
        if sys.version_info >= (3,5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
#        classes = classes[:class_nums]
        class_to_idx = {classes[i] : i for i in range(len(classes))}   
        return classes, class_to_idx
    
    def create_indices(self, dir, class_to_idx, extensions=None):
        inds = dict()
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
                        label = class_to_idx[target]
                        if label not in inds:
                            inds[label] = []
                        inds[label].append(path)                                
        return inds
    
    @staticmethod
    def generate_triplets(indices, num_triplets, n_classes):
        triplets = []
        # Indices = array of labels and each label is an array of indices
        for x in tqdm(range(num_triplets)):
            c1 = np.random.randint(0, n_classes-1)
            c2 = np.random.randint(0, n_classes-1)
            while len(indices[c1]) < 2:
                c1 = np.random.randint(0, n_classes-1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes-1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
        return triplets

    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''

        # Get the index of each image in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform images if required
        img_a, img_p, img_n = readbcn(a), readbcn(p), readbcn(n)
        #resample
        choice_a = np.random.choice(len(img_a), self.npoints, replace=False) if len(img_a) >= self.npoints else np.random.choice(len(img_a), self.npoints, replace=True)
        choice_p = np.random.choice(len(img_p), self.npoints, replace=False) if len(img_p) >= self.npoints else np.random.choice(len(img_p), self.npoints, replace=True)
        choice_n = np.random.choice(len(img_n), self.npoints, replace=False) if len(img_n) >= self.npoints else np.random.choice(len(img_n), self.npoints, replace=True)
        
        
#        choice_a, choice_p, choice_n = np.random.choice(len(img_a), self.npoints, replace=True),np.random.choice(len(img_p), self.npoints, replace=True),np.random.choice(len(img_n), self.npoints, replace=True)       
        img_a, img_p, img_n = img_a[choice_a, :], img_p[choice_p, :], img_n[choice_n, :]#choice
        
        img_a[:,0:3], img_p[:,0:3], img_n[:,0:3] = (img_a[:,0:3])/(100), (img_p[:,0:3])/(100), (img_n[:,0:3])/(100)
        img_a[:,6], img_p[:,6], img_n[:,6] = torch.pow(img_a[:,6],0.1), torch.pow(img_p[:,6],0.1), torch.pow(img_n[:,6],0.1)
        return img_a, img_p, img_n, c1, c2

    def __len__(self):
        return len(self.training_triplets)
    
if __name__ == '__main__':
    print('test')
    train_dataset = TripletFaceDataset(root = '/home/zzy/m2_500/FRGC_bcnc/fall2003',
                    n_triplets = 10000,
                    class_nums = 500,
                    transforms=None,
                    extensions='bcnc')
    print(train_dataset[0][0].shape)