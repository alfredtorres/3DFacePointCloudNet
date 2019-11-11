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
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import os.path as osp
import os
import argparse

from pointnet2.models import Pointnet2FaceClsSSG as Pointnet
from pointnet2.data import GPMMNormalCurvDataset
import pointnet2.data.data_utils as d_utils

import time
import shutil
from pointnet2.train import layer
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-model_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-cls_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=10, help="Number of epochs to train for"
    )
    # loss Classifier
    parser.add_argument('--num_class', type=int, default=500,
                    help='number of people(class)')
    parser.add_argument('--classifier_type', type=str, default='AL',
                    help='Which classifier for train. (MCP, AL, L)')

    return parser.parse_args()

lr = 1e-3
log_file = './log/train_log.txt'
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
    if epoch in [5, 8, 9]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            
def train(train_loader, model, classifier, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()

    model.train()

    end = time.time()    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # compute output
        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(-1)
        
        output = model(input_var)
        if isinstance(classifier, torch.nn.Linear):
            output = classifier(output)
        else:
            output = classifier(output, target)
        
        loss   = criterion(output, target_var)
        
        optimizer.zero_grad()
        _, classes = torch.max(output, -1)
        acc = (classes == target_var).float().sum() / target_var.numel()
        losses.update(loss.item(), input.size(0))
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            
            f.writelines('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            
            
def validate(val_loader, model, classifier, criterion):
    losses     = AverageMeter()
    top1       = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        input      = input.cuda()
        target     = target.cuda()
        input_var  = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.view(-1)
        # compute output
        output = model(input_var)
        if isinstance(classifier, torch.nn.Linear):
            output = classifier(output)
        else:
            output = classifier(output, target)
        
        loss   = criterion(output, target_var)
        # measure accuracy and record loss
        _, classes = torch.max(output, -1)
        acc = (classes == target_var).float().sum() / target_var.numel()
        losses.update(loss.item(), input.size(0))
        top1.update(acc, input.size(0))


    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))
    f.writelines('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(losses.avg, top1.avg))

    return top1.avg

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
    train_dataset = GPMMNormalCurvDataset(root = '',
                    class_nums = args.num_class,
                    transforms=transforms,
                    train=True,
                    extensions='bcnc')
    print('Train dataset Length: {}'.format(train_dataset.data_length))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataset =  GPMMNormalCurvDataset(root = '',
                    class_nums = args.num_class,
                    transforms=None,
                    train=False,
                    extensions='bcnc')
    print('Eval dataset Length: {}'.format(test_dataset.data_length))
    #print(test_set[0][0].shape)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
       [{'params': model.parameters()}, {'params': classifier.parameters()}], 
       lr=lr, weight_decay=args.weight_decay
    )

    # default value
    it = -1  
    best_prec1 = 0
    best_top1 = 0
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
    
    it = max(it, 0)  # for the initialize value of `trainer.train`

    if not osp.isdir("checkpoints"):
        os.makedirs("checkpoints")
    ## rewrite the training process
    checkpoint_name_ori="checkpoints/pointnet2_model"
    best_name="checkpoints/pointnet2_model_best"
    cls_checkpoint_name="checkpoints/pointnet2_cls"
    cls_best_name="checkpoints/pointnet2_cls_best"

    eval_frequency = len(train_loader)
    for epoch in range(args.epochs):      
        #lr_f.write()
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, classifier, criterion, optimizer, epoch)
        # evaluate on validation set
        top1, tpr = validate(test_loader, model, classifier, criterion)
        # save the learned parameters
        is_best = (top1 * tpr) > best_top1
        best_top1 = max(best_top1, top1 * tpr)
        checkpoint_name = checkpoint_name_ori + str(epoch)
        save_checkpoint(
            checkpoint_state(model, optimizer,
                             top1, args.epochs, epoch),
            is_best,
            filename=checkpoint_name,
            bestname=best_name)
    ## rewrite the training process end
    f.close()
