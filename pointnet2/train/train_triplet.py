from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable,Function
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import os.path as osp
import os
import argparse

from pointnet2.models import Pointnet2FaceClsSSG as Pointnet
from pointnet2.data import TripletFaceDataset
import pointnet2.data.data_utils as d_utils

import time
import shutil
from pointnet2.train import layer
import numpy as np
import struct 
from sklearn import metrics

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=20000, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=5e3, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-model_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-cls_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    # loss Classifier
    parser.add_argument('--margin', type=float, default=0.4, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
    parser.add_argument('--num_triplet', type=int, default=10000, metavar='num_triplet',
                    help='the margin value for the triplet loss function (default: 1e4')

    parser.add_argument('--num_class', type=int, default=500,
                    help='number of people(class)')
    parser.add_argument('--classifier_type', type=str, default='AL',
                    help='Which classifier for train. (MCP, AL, L)')

    return parser.parse_args()

lr = 1e-3
#lr_clip = 1e-5
#bnm_clip = 1e-2
log_file = './log/triplet_log.txt'
class CosineDistance(Function):
    def __init__(self, p):
        super(CosineDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        x1_norm = torch.norm(x1,2,1)
        x2_norm = torch.norm(x2,2,1)
        cos_theta = torch.sum(torch.mul(x1,x2),dim=1) / x1_norm / x2_norm
#        print(cos_theta)
        cos_theta = cos_theta.clamp(-1, 1) - 1
        return -cos_theta
    
class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
    
l2_dist = CosineDistance(2)

class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = CosineDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)
#        print(d_p)
#        print(d_n)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.6), int(args.epochs*0.9)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            
def train(train_loader, model, classifier, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    model.train()

    end = time.time()    
    for i, (data_a, data_p, data_n,label_p,label_n) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # compute output
        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a,requires_grad=True), Variable(data_p,requires_grad=True), Variable(data_n,requires_grad=True)
        
        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)
#        print(out_a.shape)
        # Choose the hard negatives
        d_p = l2_dist.forward(out_a, out_p)
        d_n = l2_dist.forward(out_a, out_n)
#        print(d_p)
#        print(d_n)

        all = (d_n - d_p < args.margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue
#        print(d_p)
        out_selected_a = out_a[hard_triplets]
#        print(out_selected_a.shape)
        out_selected_p = out_p[hard_triplets]
        out_selected_n = out_n[hard_triplets]
        
        triplet_loss = TripletMarginLoss(args.margin).forward(out_selected_a, out_selected_p, out_selected_n)
#        print(triplet_loss)
        loss   = triplet_loss
        losses.update(loss.item(), data_a.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            
            f.writelines('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))           

def loadBCNFile(path):
    n = os.path.getsize(path) // 4
    with open(path, 'rb') as f:
        data_raw = struct.unpack('f' * n, f.read(n * 4))
    data = np.asarray(data_raw, dtype=np.float32).reshape((4 + 3), n // (4 + 3))
    point_set = data.T
    # normlize
    point_set[:,0:3] = (point_set[:,0:3])/(100)
    
    point_set = torch.from_numpy(point_set)
    point_set[:,6] = torch.pow(point_set[:,6],0.1)
    return point_set

def validate(model, gfile_list, pfile_list, Label):
    model.eval()
    optimizer.zero_grad()
    # switch to evaluate mode
    g_feature = torch.zeros((len(gfile_list), 1, 512))
    p_feature = torch.zeros((len(pfile_list), 1, 512))
    
    with torch.set_grad_enabled(False):
        for i, file in enumerate(gfile_list):
            fname = file
            input = loadBCNFile(fname)
            input = input.unsqueeze(0).contiguous()
            input = input.to("cuda", non_blocking=True)
            feat = model(input)# 1x512
            g_feature[i, :, :] = feat.cpu()# 105x1x512
    #        g_label[i] = gclass_list[i]
            
        for i, file in enumerate(pfile_list):
            fname = file
            input = loadBCNFile(fname)
            input = input.unsqueeze(0).contiguous()
            input = input.to("cuda", non_blocking=True)
            feat = model(input)
            p_feature[i, :, :] = feat.cpu()# 194x1x512
    #        p_label[i] = pclass_list[i] 
    g_feature2 = torch.norm(g_feature,p=2,dim=2)
    p_feature2 = torch.norm(p_feature,p=2,dim=2)
    dis = torch.sum(torch.mul(g_feature, p_feature.transpose(1,0)), dim=2) / g_feature2 / p_feature2.transpose(1,0)
    
    top1 = np.equal(torch.argmax(dis,dim=0).numpy(), np.argmax(Label,axis=0)).sum() / len(pfile_list)
    print('\nBourphous set: top1: ({})\n'.format(top1))
    f.writelines('\nBourphous set: top1: ({})\n'.format(top1))
    
    score = dis.view(-1).numpy()
    label = np.reshape(Label,(-1))
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('Bourphous set: auc: ({})\n'.format(auc))
    f.writelines('\nBourphous set: auc: ({})\n'.format(auc))
    
    for i in range(len(fpr)):
        if fpr[i] >= 1e-3:
    #        i = i - 1
            break
    print('Bourphous tpr:{:.4f} @fpr{:.4f}\n'.format(tpr[i],fpr[i]))
    f.writelines('\nBourphous tpr:{:.4f} @fpr{:.4f}'.format(tpr[i],fpr[i]))
    return top1

def checkpoint_state(model=None,
                     optimizer=None,
                     best_prec=None,
                     epoch=None,
                     it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'it': it,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }

def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))

def get_list(folder):
    pt_list = []
    class_list = []
    gallery_classes = sorted(os.listdir(folder))
    for cname in gallery_classes:
        cpath = os.path.join(folder, cname)
        gallery = os.listdir(cpath)
        for gname in gallery:
            gpath = os.path.join(cpath, gname)
            pt_list.append(gpath)
            class_list.append(cname)
    return pt_list, class_list

if __name__ == "__main__":
    args = parse_args()
    f = open(log_file, 'w')
    transforms = transforms.Compose(
        [
            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudRotate(axis=np.array([0, 1, 0])),
            d_utils.PointcloudRotate(axis=np.array([0, 0, 1])),
            d_utils.PointcloudJitter(std=0.002),
        ]
    )
    train_dataset = TripletFaceDataset(root = '',
                    n_triplets = args.num_triplet,
                    n_points = args.num_points,
                    class_nums = 500,
                    transforms=None,
                    extensions='bcnc')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).cuda(),
        'AL' : layer.AngleLinear(512, args.num_class).cuda(),
        'L'  : torch.nn.Linear(512, args.num_class, bias=False).cuda()
    }[args.classifier_type]
    
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(
       [{'params': model.parameters()}, {'params': classifier.parameters()}], 
       lr=lr, weight_decay=args.weight_decay
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_prec1 = 0
    best_auc1 = 0
    start_epoch = 1

    # load status from checkpoint
    if args.model_checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.model_checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
    if args.cls_checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            classifier, optimizer, filename=args.cls_checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    it = max(it, 0)  # for the initialize value of `trainer.train`

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")
    ## rewrite the training process
    checkpoint_name="checkpoints/pointnet2_model"
    best_name="checkpoints/pointnet2_model_best"
    cls_checkpoint_name="checkpoints/pointnet2_cls"
    cls_best_name="checkpoints/pointnet2_cls_best"
    
    
    eval_frequency = len(train_loader)
    
    eval_gallary_floder = ''
    eval_probe_floder   = ''
    egfile_list, egclass_list= get_list(eval_gallary_floder)
    epfile_list, epclass_list= get_list(eval_probe_floder)
    elabel = np.zeros((len(egclass_list),len(epclass_list)))
    for i, gclass in enumerate(egclass_list):
        for j, eclass in enumerate(epclass_list):
            if gclass == eclass:
                elabel[i,j] = 1
            else:
                elabel[i,j] = -1
    
    for epoch in range(args.epochs):      
        #lr_f.write()
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, classifier, criterion, optimizer, epoch)
        # evaluate on validation set
        auc1 = validate(model, egfile_list, epfile_list, elabel)
        # save the learned parameters
        is_best = auc1 > best_auc1
        best_auc1 = max(best_auc1, auc1)
        save_checkpoint(
            checkpoint_state(model, optimizer,
                             auc1, args.epochs, epoch),
            is_best,
            filename=checkpoint_name,
            bestname=best_name)
    ## rewrite the training process end
    f.close()
