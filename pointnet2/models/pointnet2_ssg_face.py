from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from collections import namedtuple
#import sys
#sys.path.append('/home/zzy/file/experiment/PointNet/Pointnet2_PyTorch-master_knn')
from pointnet2.utils.pointnet2_modules import PointnetSAModule1,PointnetSAModule2


def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn


class Pointnet2SSG(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, use_xyz=True):
        super(Pointnet2SSG, self).__init__()
        print('network inital')
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule1(
                npoint=2048,
                radius=0.01,
                nsample=64,
                mlp=[input_channels, 32, 32],
                use_xyz=use_xyz,
            )
        )
        
        self.SA_modules.append(
            PointnetSAModule1(
                npoint=512,
                radius=0.12,
                nsample=64,
                mlp=[32, 64, 64],
                use_xyz=use_xyz,
            )
        )
         

        self.SA_modules.append(
            PointnetSAModule1(mlp=[64, 256, 512], use_xyz=use_xyz)
        )


        self.Feat_layer = (
            pt_utils.Seq(512)
#            .fc(512, bn=True)#, activation=None
#            .dropout(0.5)
            .fc(512, bn=False, activation=None)
        )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:6].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        carvture = pc[..., 6].contiguous() if pc.size(-1) == 7 else None

        return xyz, features, carvture

    def forward(self, pointcloud):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features, carvture = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features, carvture = module(xyz, features, carvture)

        #x = self.fc1(features.squeeze(-1))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #x = x / torch.norm(x)
        feat = self.Feat_layer(features.squeeze(-1))
        return feat#self.Shape_layer(features.squeeze(-1)), self.Exp_layer(features.squeeze(-1))#,  for extract feature to test other face datasets 

if __name__ == "__main__":
    
    from torch.autograd import Variable
    from torchviz import make_dot
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    model = Pointnet2SSG(input_channels=3, use_xyz=True).cuda()
    
    x = Variable(torch.randn(2,1024,6)).cuda()

    vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    vis_graph.view()
