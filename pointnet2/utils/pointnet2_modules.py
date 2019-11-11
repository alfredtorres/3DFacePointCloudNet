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
import sys
sys.path.append('/home/zzy/file/experiment/PointNet/Pointnet2_PyTorch-master_Curvature')
from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

class _PointnetSAModuleBase1(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase1, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None, curvature=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        
        if self.npoint is not None:
# =============================================================================
#             xyz_pairdistance = torch.topk(torch.norm(xyz,dim=2,keepdim=True),int(xyz.shape[1]*0.4),dim=1,largest=False)[1]
# #            xyz_norm = torch.norm(xyz[:,:,0:3],dim=2,keepdim=True)
# #            xyz_pairdistance = xyz_norm.lt(0.60).nonzero()[:,1].unsqueeze(0).unsqueeze(-1)
#             xyz_inner = torch.gather(xyz,1,xyz_pairdistance.repeat(1,1,3))
#             xyz_flipped = xyz_inner.transpose(1, 2).contiguous()
# 
#             curvature_inner = torch.gather(curvature.unsqueeze(-1),1,xyz_pairdistance)
#             curvature_inner = curvature_inner.squeeze(-1)
#             xyz_c = torch.stack((xyz_inner[:,:,0],xyz_inner[:,:,1],xyz_inner[:,:,2],curvature_inner.squeeze(-1)),dim=2)
# =============================================================================
            xyz_norm = torch.norm(xyz[:,:,0:3],dim=2,keepdim=True)
#            print(xyz_norm.shape)
            xyz_pairdistance = xyz_norm.squeeze(-1).gt(0.7).nonzero()
#            print(xyz_pairdistance[0:10])
#            print(xyz_pairdistance[0][1])
#            print(curvature.shape)
#            print(curvature[0:2,:])
#            print('******')
#            print(curvature[xyz_pairdistance[:,0],xyz_pairdistance[:,1]])
#            quit()
            curvature[xyz_pairdistance[:,0],xyz_pairdistance[:,1]] = 0
            xyz_flipped = xyz[:,:,0:3].transpose(1, 2).contiguous()
            xyz_c = torch.stack((xyz[:,:,0],xyz[:,:,1],xyz[:,:,2],curvature.squeeze(-1)),dim=2)           
                
            idx =  pointnet2_utils.furthest_point_sample(xyz_c.contiguous(), self.npoint)
#            idx = idx.detach()
            new_xyz = (
                pointnet2_utils.gather_operation(
                    xyz_flipped, idx
                )
                .transpose(1, 2)
                .contiguous()
                )
            new_curvature = torch.gather(curvature, dim=1, index=idx.long())
        else:
            new_xyz = None
            new_curvature = None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), new_curvature
class _PointnetSAModuleBase2(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase2, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None, curvature=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        
        if self.npoint is not None:
            xyz_pairdistance = torch.topk(torch.norm(xyz,dim=2,keepdim=True),int(xyz.shape[1]*0.4),dim=1,largest=False)[1]
#            xyz_norm = torch.norm(xyz[:,:,0:3],dim=2,keepdim=True)
#            xyz_pairdistance = xyz_norm.lt(0.40).nonzero()[:,1].unsqueeze(0).unsqueeze(-1)
            xyz_inner = torch.gather(xyz,1,xyz_pairdistance.repeat(1,1,3))
            xyz_flipped = xyz_inner.transpose(1, 2).contiguous()

            curvature_inner = torch.gather(curvature.unsqueeze(-1),1,xyz_pairdistance)
            curvature_inner = curvature_inner.squeeze(-1)
            xyz_c = torch.stack((xyz_inner[:,:,0],xyz_inner[:,:,1],xyz_inner[:,:,2],curvature_inner.squeeze(-1)),dim=2)
     
            idx =  pointnet2_utils.furthest_point_sample(xyz_c.contiguous(), self.npoint)
            idx = idx.detach()
            new_xyz = (
                pointnet2_utils.gather_operation(
                    xyz_flipped, idx
                )
                .transpose(1, 2)
                .contiguous()
                )
            new_curvature = torch.gather(curvature, dim=1, index=idx.long())
        else:
            new_xyz = None
            new_curvature = None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1), new_curvature


class PointnetSAModuleMSG2(_PointnetSAModuleBase2):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG2, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule2(PointnetSAModuleMSG2):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule2, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )





class PointnetSAModuleMSG1(_PointnetSAModuleBase1):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG1, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule1(PointnetSAModuleMSG1):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule1, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_curvature = Variable(torch.randn(2, 9, 1).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats, xyz_curvature))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

# =============================================================================
#     for _ in range(1):
#         _, new_features,_ = test_module(xyz, xyz_feats)
#         new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
#         print(new_features)
#         print(xyz.grad)
# =============================================================================
