"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from typing import List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import torch.jit as jit
import kornia
from torch.cuda import device

from Losses.custom_functional import *
import Losses.ChamferLoss.chamfer2D.dist_chamfer_2D as dist_chamfer_2D
from Losses.custom_functional import compute_grad_mag
from Losses.df_loss import EuclideanLossWithOHEM
from matplotlib import pyplot as plt
from Losses.abl_ori import ABL
from config import cfg
from Losses.DualTaskLoss import DualTaskLoss
from Losses import pytorch_ssim
from Losses.HarrisCorner import CornerDetection, NonMaxSuppression


def get_loss(args):
    '''
    Get the criterion based on the loss function
    args: 
    return: criterion
    '''

    if args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=args.dataset_cls.num_classes, reduction=True,
            ignore_index=args.dataset_cls.ignore_label,
            upper_bound=args.wt_bound).cuda()
    elif args.joint_edgeseg_loss:
        criterion = JointEdgeSegLoss(classes=args.dataset_cls.num_classes,
                                     ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
                                     edge_weight=args.edge_weight, seg_weight=args.seg_weight,
                                     att_weight=args.att_weight, dual_weight=args.dual_weight).cuda()

    else:
        criterion = CrossEntropyLoss2d(reduction=True,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    criterion_val = JointEdgeSegLoss(classes=args.dataset_cls.num_classes, mode='val',
                                     ignore_index=args.dataset_cls.ignore_label, upper_bound=args.wt_bound,
                                     edge_weight=args.edge_weight, seg_weight=args.seg_weight).cuda()

    return criterion, criterion_val


def _iou(pred, target):
    b = pred.shape[0]
    pred = torch.sigmoid(pred)
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        return _iou(pred, target)


class BCE_IoULoss(torch.nn.Module):
    def __init__(self):
        super(BCE_IoULoss, self).__init__()
        self.iou_loss = IoULoss()
        self.bce = BCELoss()

    def forward(self, pred, target):
        iou = self.iou_loss(pred, target)
        bce = self.bce(pred, target)
        return iou + bce


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, deep_supervision, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train',
                 edge_weight=200, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if deep_supervision:
            self.seg_loss = DeepSupervisionLoss(typeloss='CELoss')
        elif mode == 'train':
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(reduction=True,
                                               ignore_index=ignore_index).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss()

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss

    def edge_attention(self, input, target, edge):
        # n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        print('edge attention target size:', target.shape)
        return self.seg_loss(input,
                             torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        losses = {}

        losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        losses['edge_loss'] = self.edge_weight * self.bce2d(edgein, edgemask)
        # losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)  # 把target中edge部分提取出来计算损失
        # losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)

        return losses


# Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, ignore_index, reduction=reduction)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), density=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0), dim=1),
                                  targets[i])
        return loss


# Cross Entropy NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, ignore_index, reduction=reduction)

    def forward(self, inputs, targets):
        # print('Nllloss targets shape: ', targets.shape)
        return self.nll_loss(F.log_softmax(inputs), targets)


class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        # print(pred_flat.shape, "   ", target_flat.shape)
        # pred_flat = torch.sigmoid(pred_flat)
        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1e-5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        pred = F.sigmoid(pred)

        # flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


""" Deep Supervision Loss"""


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.seg_loss = BCELoss()
        self.seg_edge_loss = ABL()

    def bce2d(self, input, target):
        n, c, h, w = input.size()
        # print('target mim max', target.min(), target.max())
        # print('bce  input shape:', input.shape, ' target shape:', target.shape)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()
        # print('target_t:', target_t.shape)
        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss

    def forward(self, pred, gt):
        pred_segs, pred_edges = pred
        gt_seg, gt_edge = gt
        gt_seg, gt_edge = gt_seg.to(torch.float32), gt_edge.to(torch.float32)
        size = gt_seg.size()[-2:]
        seg_loss = 0
        edge_loss = 0
        for seg in pred_segs:
            seg = F.interpolate(seg, size=size, mode='bilinear', align_corners=True)
            loss = self.seg_loss(seg, gt_seg)
            seg_loss = seg_loss + loss
        for edge in pred_edges:
            edge = F.interpolate(edge, size=size, mode='bilinear', align_corners=True)
            loss = self.bce2d(edge, gt_edge)
            edge_loss += loss
        # seg_edge_loss = self.seg_edge_loss(pred_segs[0], gt_edge)
        seg_edge_loss = torch.tensor(0.0, device=pred_segs[0].device)
        return {'seg_loss': seg_loss, 'edge_loss': edge_loss, 'seg_edge_loss': seg_edge_loss}


class JointEdgeDeepSupervisionLoss(nn.Module):
    def __init__(self, seg_weight=1, edge_weight=10):
        super(JointEdgeDeepSupervisionLoss, self).__init__()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
        self.criterion_seg = DeepSupervisionLoss('BceIoULoss')
        self.criterion_edge = self.bce2d

    def bce2d(self, input, target):
        n, c, h, w = input.size()
        # print('input size', input.shape,'target size', target.shape)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # print('log p size: ',log_p.size())
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # print('target_t:', target_t.size())
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss

    def forward(self, inputs, targets):
        seg_preds, edge_pred = inputs
        seg_gt, edge_gt = targets
        seg_gt = seg_gt.to(torch.float32)
        seg_loss = 0
        for seg in seg_preds:
            loss = self.bce2d(seg, seg_gt)
            seg_loss = seg_loss + loss
        edge_loss = self.criterion_edge(edge_pred, edge_gt)
        losses = {}
        losses['seg_loss'] = seg_loss * self.seg_weight
        losses['edge_loss'] = edge_loss * self.edge_weight
        return losses


bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, reduction='mean')


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        device = predict.device
        target = target.contiguous().view(target.shape[0], -1)
        target_gpu = target.clone().cuda(device=device)
        valid_mask_gpu = valid_mask.clone().cuda(device=device) if valid_mask is not None else torch.ones_like(target_gpu,device=device)
        valid_mask_gpu = valid_mask_gpu.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_gpu) * valid_mask_gpu, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target_gpu.pow(self.p)) * valid_mask_gpu, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class CertainLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CertainLoss, self).__init__()
        self.reduction = reduction

    def forward(self, uncertain):
        loss = 0
        for u in uncertain:
            loss += torch.mean(u)
        return loss

def convert_to_one_hot(x, minleng, ignore_idx=-1):
    """
    encode input x into one hot
    inputs:
        x: tensor of shape (N, ...) with type long
        minleng: minimum length of one hot code, this should be larger than max value in x
        ignore_idx: the index in x that should be ignored, default is 255

    return:
        tensor of shape (N, minleng, ...) with type float
    """
    device = x.device
    # compute output shape
    size = list(x.size())
    size.insert(1, minleng)
    assert x[x != ignore_idx].max() < minleng, "minleng should larger than max value in x"

    if ignore_idx < 0:
        out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
    else:
        # overcome ignore index
        with torch.no_grad():
            x = x.clone().detach()
            ignore = x == ignore_idx
            x[ignore] = 0
            out = torch.zeros(size, device=device).scatter_(1, x.unsqueeze(1), 1)
            ignore = ignore.nonzero(as_tuple=False)
            _, M = ignore.size()
            a, *b = ignore.chunk(M, dim=1)
            out[[a, torch.arange(minleng), *b]] = 0
    return out


class AffinityLoss(nn.Module):

    def __init__(self, kernel_size=3, ignore_index=-100):
        super(AffinityLoss, self).__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self.unfold = nn.Unfold(kernel_size=kernel_size)
        # self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, logits, labels):
        """
        usage similar to nn.CrossEntropyLoss:
            >>> criteria = AffinityLoss(kernel_size=3, ignore_index=255)
            >>> logits = torch.randn(8, 19, 384, 384) # nchw
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw
            >>> loss = criteria(logits, lbs)
        """
        context_size = self.kernel_size * self.kernel_size
        logits = torch.sigmoid(logits)  # for binary classification
        logits = torch.cat([1 - logits, logits], dim=1)
        labels = torch.cat([1 - labels, labels], dim=1)
        n, c, h, w = logits.size()
        logits_unfold = self.unfold(logits).view(n, c, context_size, -1)
        lbs_unfold = self.unfold(labels).view(n, c, context_size, -1)
        aff_map = torch.einsum('ncal,ncbl->nabl', logits_unfold, logits_unfold)
        lb_map = torch.einsum('ncal,ncbl->nabl', lbs_unfold, lbs_unfold)
        loss = self.bce(aff_map, lb_map)
        return loss


class RefineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss()
        self.dice = BinaryDiceLoss()

    def forward(self, pred, gt, valid_mask=None):
        bce_loss = self.bce(pred, gt)
        dice_loss = self.dice(pred, gt, valid_mask)
        return bce_loss + dice_loss

def extract_edges(pred):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    if pred.is_cuda:
        sobel_x, sobel_y = sobel_x.cuda(), sobel_y.cuda()
    grad_x = F.conv2d(pred, sobel_x, padding=1)
    grad_y = F.conv2d(pred, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2)

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = pytorch_ssim.SSIM(window_size=7, reduction='mean')
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, boundaries: List[torch.Tensor], segmentations: List[torch.Tensor], gt_seg, weights): # boundaries和segmentations都是list，长度相等，各元素相对应
        assert len(boundaries) == len(segmentations), 'The length of boundaries and segmentations should be equal.'
        loss = 0
        cuda = boundaries[0].is_cuda
        size = gt_seg.size()[2:]
        for i,(boundary, segmentation) in enumerate(zip(boundaries, segmentations)):
            assert boundary.size() == segmentation.size(), 'The size of boundary and segmentation should be equal.'
            #size = segmentation.size()[2:]
            segmentation_sig = torch.sigmoid(segmentation)
            boundary_sig = torch.sigmoid(boundary)

            #print('gt_edge_now shape:', gt_seg.shape)
            mask = torch.zeros_like(gt_seg,dtype=segmentation_sig.dtype)
            mask[:,:, 4:-4, 4:-4] = 1.0
            #print(mask[0][0])
            #segmentation_sig = F.interpolate(segmentation_sig, size=size, mode='bilinear', align_corners=True)
            seg_boundary = compute_grad_mag(segmentation_sig,cuda=cuda) * mask
            seg_boundary = seg_boundary / (seg_boundary.max() + 1e-6)
            grad_gt_seg = compute_grad_mag(gt_seg, cuda=cuda) * mask
            grad_gt_seg = grad_gt_seg / (grad_gt_seg.max() + 1e-6)

            #boundary_sig = F.interpolate(boundary_sig, size=size, mode='bilinear', align_corners=True)
            boundary_sig = boundary_sig * mask
            boundary_sig = convTri(boundary_sig, 4, cuda)
            '''np_boundary_sig = boundary_sig.clone()[0].permute(1,2,0).cpu().detach().numpy()
            np_seg_boundary = seg_boundary.clone()[0].permute(1,2,0).cpu().detach().numpy()
            np_seg = segmentation_sig.clone()[0].permute(1,2,0).cpu().detach().numpy()
            plt.subplot(1,3,1)
            plt.imshow(np_boundary_sig)
            plt.subplot(1,3,2)
            plt.imshow(np_seg_boundary)
            plt.subplot(1,3,3)
            plt.imshow(grad_gt_seg[0].permute(1,2,0).cpu().detach().numpy())
            plt.show()'''
            loss += weights[i]*(self.l1(seg_boundary, grad_gt_seg) + self.l1(boundary_sig, grad_gt_seg))#+\
            #(1 - self.ssim(boundary_sig, seg_boundary))
        return loss

class DeepConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = pytorch_ssim.SSIM(window_size=7, reduction='mean')

    def forward(self, inputs):
        seg, edge = inputs
        seg, edge = torch.sigmoid(seg), torch.sigmoid(edge)
        loss = 1-(1+self.ssim(seg, edge))/2

        return loss

class DeepSupervisionCELoss(nn.Module):
    def __init__(self, seg_weight=1, edge_weight=5, corner_weight=0.03, dist_weight=1, aff_weight=1.5,point_weight=5,
                 use_iou=True,
                 deep_supervision=True,
                 consistency = 0,
                 corner_points=False,
                 distance=False, affinity=False,point_loss=False):
        super(DeepSupervisionCELoss, self).__init__()
        self.shape_loss = None
        self.iou_loss = use_iou
        self.ds = deep_supervision
        self.use_corner_points = corner_points
        self.use_distance = distance
        self.use_affinity = affinity
        self.use_point_seg = point_loss
        self.use_shape = 0  # 默认是0，且不能在__init__中设置，只能在外部调用set_fourier_weight时设置
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight
        self.corner_weight = corner_weight
        self.dist_weight = dist_weight
        self.affinity_weight = aff_weight
        self.point_seg_weight = point_weight
        self.consistency_weight = consistency
        self.seg_loss = BCE_IoULoss() if self.iou_loss else BCELoss() #
        self.dice = BinaryDiceLoss()
        self.dist_loss = ABL()
        self.point_loss = PointSegLoss()
        self.dir_loss = EuclideanLossWithOHEM()
        self.chamfer_loss = CornerPointsDistLoss()
        self.affinity_loss = AffinityLoss(kernel_size=3)
        self.certainty_loss = CertainLoss()
        self.consistency_loss = ConsistencyLoss()
        # self.shape_loss = FourierLoss(fourier_shape)

    def set_fourier_descriptor(self, fourier_shape):  # 只有在外部调用set weight时才会使用fourier_loss
        self.use_shape = fourier_shape
        self.shape_loss = FourierLoss(fourier_shape)

    def introduce_corner(self):
        self.use_corner_points = True
        print('Corner points introduced!')

    def bce2d(self, input, target):
        n, c, h, w = input.size()
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / (sum_num+1e-6)
        weight[neg_index] = pos_num * 1.0 / (sum_num+1e-6)

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss

    def forward(self, pred, gt):
        pred_segs = pred['seg_pred']
        aux_segs = pred['aux_pred'] if 'aux_pred' in pred else None
        pred_edges = pred['edge_pred'] if 'edge_pred' in pred else None
        pred_point_logits = pred['point_pred'] if 'point_pred' in pred else None
        uncertainty = pred['uncertain'] if 'uncertain' in pred else None
        gt_seg, gt_edge = gt[:2] if self.use_distance else gt
        gt_dist = gt[2] if self.use_distance else None
        gt_seg = gt_seg.to(torch.float32)
        size = gt_seg.size()[-2:]
        seg_loss, edge_loss, points_loss, dist_loss, affinity_loss, point_loss, consistency_loss,aux_loss = 0., 0., 0., 0., .0,.0,.0,.0
        certainty_loss = 0
        weights = [1, 1, 0.5, 0.5, 0.3, 0.3]
        #weights = [1,0.25,0.25,0.25,0.25]
        if (pred_edges is not None) and self.ds:
            for i in range(len(pred_edges)):
                pred_edges[i] = F.interpolate(pred_edges[i], size=size, mode='bilinear', align_corners=True)

        if (pred_segs is not None) and self.ds:
            for i in range(len(pred_segs)):
                pred_segs[i] = F.interpolate(pred_segs[i], size=size, mode='bilinear', align_corners=True)

        if self.ds:
            for idx, seg in enumerate(pred_segs):
                if idx == 0 and self.use_shape > 0:
                    fourier_weight = self.shape_loss(seg, gt_seg, pred['weight'])
                    stage_loss = self.seg_loss(seg, gt_seg) + fourier_weight * 0.5
                else:
                    # seg = F.interpolate(seg, size=size, mode='bilinear', align_corners=True)
                    #gt_seg_now = F.interpolate(gt_seg, size=seg.size()[2:], mode='bilinear', align_corners=True)
                    stage_loss = self.seg_loss(seg, gt_seg)
                seg_loss = seg_loss + stage_loss * weights[idx]
                #seg_loss = seg_loss * weights[idx] + stage_loss
        else:
            seg_loss = self.seg_loss(pred_segs[0], gt_seg)
        if aux_segs is not None:
            #print('calculating aux loss')
            for idx, seg in enumerate(aux_segs):
                gt_seg_now = F.interpolate(gt_seg, size=seg.size()[2:], mode='bilinear', align_corners=False)
                stage_loss_ = self.seg_loss(seg, gt_seg_now)
                aux_loss = aux_loss + stage_loss_ * weights[idx]
        if self.edge_weight>0:
            if self.ds:
                if pred_edges is not None:
                    for idx, edge in enumerate(pred_edges):
                        #gt_edge_now = F.interpolate(gt_edge, size=edge.size()[2:], mode='bilinear',align_corners=True) if idx > 0 else gt_edge
                        #print(edge.shape, gt_edge_now.shape)
                        bce = self.bce2d(edge,gt_edge)
                        #dice = self.dice(edge,gt_edge_now,None)
                        edge_loss = bce * weights[idx] + edge_loss
            else:
                edge_loss = self.bce2d(pred_edges[0], gt_edge)
        if pred_point_logits is not None:
            assert pred['point_coords'] is not None, 'Point coordinates are not provided!'
            point_coords = pred['point_coords']
            point_loss+=self.point_loss(pred_point_logits, gt_seg, point_coords)
        if self.use_corner_points:
            points_loss = self.chamfer_loss(pred_segs[0], gt_seg) + points_loss
        if self.use_distance:
            dist_loss_now = self.dist_loss(pred_segs[0], gt_dist)
            if dist_loss_now is not None:
                dist_loss = dist_loss_now + dist_loss
        if self.use_affinity:
            affinity_loss += self.affinity_loss(pred_segs[0], gt_seg)
        if uncertainty is not None:
            certainty_loss = self.certainty_loss(uncertainty)
        if self.consistency_weight>0:
            assert pred_edges is not None, 'Edge prediction is not provided!'
            cweights = [0.5,.8,1,1]
            consistency_loss += self.consistency_loss(pred_edges, pred_segs, gt_seg,cweights)
        '''assert not torch.isnan(seg_loss).any(), "NaN detected in seg loss"
        assert not torch.isnan(edge_loss).any(), "NaN detected in edge loss"
        assert not torch.isnan(aux_loss).any(), "NaN detected in aux loss"'''
        loss = seg_loss * self.seg_weight + edge_loss * self.edge_weight + points_loss * self.corner_weight \
               + dist_loss * self.dist_weight + affinity_loss * self.affinity_weight + point_loss * self.point_seg_weight\
                + certainty_loss + consistency_loss * self.consistency_weight+aux_loss
        #print('seg loss:', seg_loss, 'edge loss:', edge_loss, 'consistency loss:', consistency_loss)
        return loss


class PointSegLoss(nn.Module):
    def __init__(self):
        super(PointSegLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, point_logits, target, coords):
        loss = 0
        with torch.autograd.set_detect_anomaly(True):
            for logits, coords in zip(point_logits,coords):
                coords = coords.unsqueeze(2)*2.0-1.0
                point_label = F.grid_sample(target, coords).squeeze(3)  # (n,1,p)
                loss+=self.loss(logits, point_label)
        return loss

class PointClassifyLoss(nn.Module):
    def __init__(self):
        super(PointClassifyLoss, self).__init__()
        self.loss = nn.BCELoss()

    @staticmethod
    def point_sample_3D(input, point_indices):  # with batch dim   point_indices:N,n 其中的值为
        bs, c, h, w = input.shape
        N, n = point_indices.shape
        point_indices = point_indices.unsqueeze(1).expand(-1, c, -1)  # N,c,n
        max_index = bs * h * w
        invalid_mask = point_indices > max_index
        point_indices[invalid_mask] = 0
        flatten_input = input.permute(1, 0, 2, 3).contiguous().view(c, -1)  # c,h*w*bs
        flatten_input = flatten_input.unsqueeze(0).expand(N, -1, -1)  # N,c,h*w*bs
        sampled_feats = flatten_input.gather(dim=2, index=point_indices).view(N, c, n)
        return sampled_feats

    def forward(self, pred_points, pred_coordinate, gt_mask):  # gt_mask: (batch_size,1, 256, 256)
        batch_size, width = gt_mask.shape[0], gt_mask.shape[-1]
        loss = 0
        for i, (points_seg, coords) in enumerate(zip(pred_points, pred_coordinate)):
            coords[:, :, 1:] = coords[:, :, 1:] * (2 ** i)
            index = (coords[..., 0] * width * width + coords[..., 1] * width + coords[..., 2]).long()
            gathered_gt = self.point_sample_3D(gt_mask, index).to(torch.float32)
            # print('gathered gt type:', gathered_gt.dtype, 'seg type:', points_seg.dtype)
            loss += self.loss(points_seg, gathered_gt)
        return loss


class JointDecoupleCascadeLoss(nn.Module):
    def __init__(self):
        super(JointDecoupleCascadeLoss, self).__init__()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.edge_loss = self.bce2d
        self.body_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, gts):
        size = gts[0].size()[2:]
        # print(size)
        seg_loss, edge_loss, body_loss = 0, 0, 0
        weights = [1, 0.5, 0.5, 0.3, 0.3]
        for idx, seg in enumerate(preds['seg_pred']):
            seg = F.interpolate(seg, size=size, mode='bilinear', align_corners=True)
            seg_loss += self.seg_loss(seg, gts[0]) * weights[idx]
        for idx, body in enumerate(preds['body_pred']):
            body = F.interpolate(body, size=size, mode='bilinear', align_corners=True)
            body_loss += self.body_loss(body, gts[2]) * weights[idx + 1]
        for idx, edge in enumerate(preds['edge_pred']):
            edge = F.interpolate(edge, size=size, mode='bilinear', align_corners=True)
            edge_loss += self.edge_loss(edge, gts[1]) * weights[idx + 1]
        return body_loss + seg_loss + edge_loss

    def bce2d(self, input, target):
        n, c, h, w = input.size()
        # print('input size', input.shape,'target size', target.shape)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        # print('log p size: ',log_p.size())
        # target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.contiguous().view(1, -1)
        # print('target_t:', target_t.size())
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
        return loss


class CornerPointsDistLoss(nn.Module):
    def __init__(self):
        super(CornerPointsDistLoss, self).__init__()
        self.eps = 1e-6
        self.corner_det = CornerDetection(corner_window=3, nms_window=5)
        self.chamfer = dist_chamfer_2D.chamfer_2DDist()

    def forward(self, pred, target):
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        pred = (torch.sigmoid(pred) > 0.5).float()
        pred_corner = self.corner_det(pred)
        target_corner = self.corner_det(target)
        chamfer_loss = 0.
        for b in range(pred_corner.shape[1]):
            pred_eig_val = pred_corner[1, b]
            gt_eig_val = target_corner[1, b]
            pred_corner_coords = torch.nonzero(pred_eig_val).unsqueeze(0)
            if pred_corner_coords.numel() == 0:
                # print('empty pred!')
                continue
            gt_corner_coords = torch.nonzero(gt_eig_val).unsqueeze(0)
            if gt_corner_coords.numel() == 0:
                # print('empty gt!')
                continue
            dist1, dist2, _, _ = self.chamfer(gt_corner_coords.float(), pred_corner_coords.float())
            dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
            dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
            dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
            if torch.isnan(dist).any():
                # print(dist1,'\n',dist2)
                continue
            # print('dist1 mean:',dist1.mean(-1),'dist2 mean:',dist2.mean(-1),'dist:', dist)
            chamfer_loss = chamfer_loss + dist
        return chamfer_loss


@jit.script
def sort_points(points):
    """根据极角对点进行排序"""
    center = points.float().mean(dim=0)
    angles = torch.atan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return center, points[torch.argsort(angles)]


@jit.script
def calculate_fourier_coefficients(k: int, l, delta):
    # print('In calculate_fourier_coefficients, l shape:', len(l))
    L = l[-1]
    non_zero_delta = delta != 0
    a = torch.sum(delta[non_zero_delta] * torch.sin((2 * torch.pi * k * l[non_zero_delta]) / L)) / (k * torch.pi)
    b = -torch.sum(delta[non_zero_delta] * torch.cos((2 * torch.pi * k * l[non_zero_delta]) / L)) / (k * torch.pi)
    return torch.sqrt(a * a + b * b)


@jit.script
def findContours(I):  # I: 2D image 预测或者真实的mask, tensor shape: (B, 1, H, W), sigmoid之后的值
    device = I.device
    # print('In findContours, I shape:', I.shape)
    I = F.pad(I, (1, 1, 1, 1), mode='constant', value=float(0))
    connected_components = kornia.contrib.connected_components(I, num_iterations=500)[:, :, 1:-1, 1:-1]
    kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], device=device).float().unsqueeze(0).unsqueeze(0)
    boundary_mask = F.conv2d(I, kernel, padding=1)
    boundary_mask = (boundary_mask < 0).float()[:, :, 1:-1, 1:-1]  # 边界点值为1
    contour = boundary_mask * connected_components
    # print('contour:', contour.shape)
    return contour


@jit.script
def calculate_fourier_descriptors2(contour, center, N: int, weight: torch.Tensor) -> torch.Tensor:
    # print('In calculate_fourier_descriptors, contour shape:', contour.shape)
    points1 = contour
    points2 = torch.roll(contour, shifts=-1, dims=0)

    d1 = torch.sqrt((points1[:, 1] - center[1]) ** 2 + (points1[:, 0] - center[0]) ** 2)
    d2 = torch.sqrt((points2[:, 1] - center[1]) ** 2 + (points2[:, 0] - center[0]) ** 2)
    delta = d1 - d2

    d3 = torch.sqrt((points1[:, 1] - points2[:, 1]) ** 2 + (points1[:, 0] - points2[:, 0]) ** 2)
    l = torch.cumsum(d3, dim=0)

    A = torch.empty(N, dtype=torch.float32, device=contour.device)
    for i in range(1, N + 1):
        A[i - 1] = calculate_fourier_coefficients(i, l, delta)
    # print('A:',A.shape)
    return A * (1 / 2 * (weight ** 2)) + torch.log(1 + weight ** 2)
    # return A * weight


@jit.script
def match(target, source):  # 对batch里的同一个样本进行物体匹配 centers都是至少有一个物体的 shape: (n,2)
    # print('In match, target shape:', target.shape, 'source shape:', source.shape)
    dist_matrix = torch.cdist(target, source)
    matching_idx = torch.argmin(dist_matrix, dim=1)
    return matching_idx


@jit.script
def calculate_regularization(pred_contours: torch.Tensor, target_contours: torch.Tensor,
                             N: int,
                             weight: torch.Tensor) -> torch.Tensor:  # contours from one batch, gt or pred  shape: b,1,h,w
    # print('In calculate_regularization, pred_contours shape:', pred_contours.shape, 'target_contours shape:', )
    assert pred_contours.shape[0] == target_contours.shape[0], "batch size not match"
    total_diff = torch.tensor(0.0, device=pred_contours.device)
    for b in range(pred_contours.shape[0]):
        pred_descriptors_sample, pred_centers_sample, gt_descriptors_sample, gt_centers_sample = [], [], [], []
        pred_batch_contour = pred_contours[b][0]  # shape: (1, H, W)  一张图有若干个contour
        target_batch_contour = target_contours[b][0]
        # print('target:', target_batch_contour.sum(), 'pred:', pred_batch_contour.sum())
        if (target_batch_contour.sum() == 0) or (pred_batch_contour.sum() == 0):
            continue
        unique_values = torch.unique(pred_batch_contour)
        for building in unique_values:
            contour_points = torch.nonzero(pred_batch_contour == building)  # shape: (n, 2) n为contour上的点数，2为坐标
            center, contour_points = sort_points(contour_points)  # 按照极角排序  center是中心点坐标，用于匹配
            pred_centers_sample.append(center)
            fourier_descriptors = calculate_fourier_descriptors2(contour_points, center, N, weight)
            # print('descriptors device', fourier_descriptors.device, 'center device:', center.device)
            pred_descriptors_sample.append(fourier_descriptors)
        target_unique_values = torch.unique(target_batch_contour)
        for building in target_unique_values:
            contour_points = torch.nonzero(target_batch_contour == building)
            center, contour_points = sort_points(contour_points)
            gt_centers_sample.append(center)
            fourier_descriptors = calculate_fourier_descriptors2(contour_points, center, N, weight)
            gt_descriptors_sample.append(fourier_descriptors)
        # center[0]是物体较少的，center[1]是物体较多的
        if len(gt_centers_sample) <= len(pred_centers_sample):
            matching = match(torch.stack(gt_centers_sample), torch.stack(pred_centers_sample))
            descriptor_diff = torch.sum(  # 原本是mean
                torch.abs(torch.stack(pred_descriptors_sample)[matching] - torch.stack(gt_descriptors_sample)), dim=1)
            # print('descriptor_diff:', descriptor_diff.shape)
        else:
            matching = match(torch.stack(pred_centers_sample), torch.stack(gt_centers_sample))
            descriptor_diff = torch.sum(
                torch.abs(torch.stack(gt_descriptors_sample)[matching] - torch.stack(pred_descriptors_sample)), dim=1)
            # print('descriptor_diff:', descriptor_diff.shape)
        total_diff += torch.sum(descriptor_diff)
    return total_diff


class FourierLoss(nn.Module):
    def __init__(self, N=2):
        super(FourierLoss, self).__init__()
        self.N = N
        # print('Fourier loss initialized!')
        # self.weight = nn.Parameter(torch.tensor((3.,1.),device='cuda'),requires_grad=True)

    def forward(self, pred, target, weight):
        if self.N == 0:
            return torch.tensor(0.0, device=pred.device)
        # print('In FourierLoss forward, pred shape:', pred.shape, 'target shape:', target.shape)
        pred = (torch.sigmoid(pred) > 0.5).float()
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        # time1 = time.time()
        pred_contours = findContours(pred)
        # time2 = time.time()
        target_contours = findContours(target)
        # time3 = time.time()
        # print('findContours time:', time2 - time1, time3 - time2)
        # weight = torch.tensor((weight,1-weight),device=pred.device)
        regularization = calculate_regularization(pred_contours, target_contours, self.N, weight)
        # print('weight:',weight)
        # time4 = time.time()
        # print('calculate_regularization time:', time4 - time3)
        return regularization


@jit.script
def calculate_fourier_coefficients_simple(k: int, contour):
    """只取边界点的x,y坐标，做傅里叶变换"""
    factor = len(contour)
    x, y = contour[:, 0], contour[:, 1]
    complex_coord = x + 1j * y
    fourier = torch.fft.fft(complex_coord)
    a = torch.real(fourier).sum() / (k * torch.pi)
    b = torch.imag(fourier).sum() / (k * torch.pi)
    return torch.sqrt(a * a + b * b) / factor


@jit.script
def calculate_fourier_descriptors_simple(contour, N: int, weight) -> torch.Tensor:
    A = torch.empty(N, dtype=torch.float32, device=contour.device)
    for i in range(1, N + 1):
        A[i - 1] = calculate_fourier_coefficients_simple(i, contour)
    return A * (1 / 2 * (weight ** 2)) + torch.log(1 + weight ** 2)


class SimpleFourierLoss(nn.Module):
    def __init__(self, N=2):
        super(SimpleFourierLoss, self).__init__()
        self.N = N

    @staticmethod
    @jit.script
    def calculate_regularization(pred_contours: torch.Tensor, target_contours: torch.Tensor,
                                 N: int,
                                 weight: torch.Tensor) -> torch.Tensor:  # contours from one batch, gt or pred  shape: b,1,h,w
        # print('In calculate_regularization, pred_contours shape:', pred_contours.shape, 'target_contours shape:', )
        assert pred_contours.shape[0] == target_contours.shape[0], "batch size not match"
        total_diff = torch.tensor(0.0, device=pred_contours.device)
        for b in range(pred_contours.shape[0]):
            # print('b:', b)
            pred_descriptors_sample, pred_centers_sample, gt_descriptors_sample, gt_centers_sample = [], [], [], []
            pred_batch_contour = pred_contours[b][0]  # shape: (1, H, W)  一张图有若干个contour
            target_batch_contour = target_contours[b][0]
            '''plt.subplot(121)
            plt.imshow(pred_batch_contour.cpu().numpy())
            plt.subplot(122)
            plt.imshow(target_batch_contour.cpu().numpy())
            plt.show()'''
            unique_values = torch.unique(pred_batch_contour)
            target_unique_values = torch.unique(target_batch_contour)
            # print('unique values ',len(unique_values),' target unique values ', len(target_unique_values))
            # print('target:', target_batch_contour.sum(), 'pred:', pred_batch_contour.sum())
            if (len(unique_values) == 1) or (len(target_unique_values) == 1):
                # print('empty contour!')
                continue
            for building in unique_values:
                # print('pred building:', building)
                contour_points = torch.nonzero(pred_batch_contour == building)  # shape: (n, 2) n为contour上的点数，2为坐标
                center, contour_points = sort_points(contour_points)  # 按照极角排序  center是中心点坐标，用于匹配
                if contour_points.size()[0] < 10:
                    continue
                # print('contourpoints shape:', contour_points.shape)
                pred_centers_sample.append(center)
                fourier_descriptors = calculate_fourier_descriptors_simple(contour_points, N, weight)
                # print('descriptors device', fourier_descriptors.device, 'center device:', center.device)
                pred_descriptors_sample.append(fourier_descriptors)

            for building in target_unique_values:
                # print('gt building:', building)
                contour_points = torch.nonzero(target_batch_contour == building)
                center, contour_points = sort_points(contour_points)
                if contour_points.size()[0] < 10:
                    continue
                # print('gt contourpoints shape:', contour_points.shape)
                gt_centers_sample.append(center)
                fourier_descriptors = calculate_fourier_descriptors_simple(contour_points, N, weight)
                gt_descriptors_sample.append(fourier_descriptors)
            # center[0]是物体较少的，center[1]是物体较多的
            if len(gt_centers_sample) <= len(pred_centers_sample):
                matching = match(torch.stack(gt_centers_sample), torch.stack(pred_centers_sample))
                weighted_descriptor_diff = torch.abs(
                    torch.stack(pred_descriptors_sample)[matching] - torch.stack(gt_descriptors_sample))
                descriptor_diff = torch.sum(weighted_descriptor_diff, dim=1)
            else:
                matching = match(torch.stack(pred_centers_sample), torch.stack(gt_centers_sample))
                weighted_descriptor_diff = torch.abs(
                    torch.stack(gt_descriptors_sample)[matching] - torch.stack(pred_descriptors_sample))
                descriptor_diff = torch.sum(weighted_descriptor_diff, dim=1)
            # print('descriptor diff : ',descriptor_diff)
            total_diff += torch.sum(descriptor_diff)
        return total_diff

    def forward(self, pred_seg, target_label, descriptors_weight):
        if self.N == 0:
            return torch.tensor(0.0, device=pred_seg.device)
        pred_seg = (torch.sigmoid(pred_seg) > 0.5).float()
        # pred_seg= (pred_seg>0.5).float()
        # print('dtype:', pred_seg.dtype, target_label.dtype)
        if len(target_label.shape) == 3:
            target_label = target_label.unsqueeze(1)
        pred_contours = findContours(pred_seg)
        target_contours = findContours(target_label)
        regularization = self.calculate_regularization(pred_contours, target_contours, self.N, descriptors_weight)
        return regularization


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import ChamferLoss.chamfer2D.dist_chamfer_2D
    from PIL import Image
    from Model.UnetASPP import ASPPUNet
    from torchvision.transforms.functional import to_tensor

    label_path = r'D:\ztb\DeepLearning\EdgeAlign\examples'
    labels = ['val_534.tif_2.png', 'val_535.tif_2.png', 'val_535.tif_3.png', 'val_535.tif_4.png']
    preds = ['val_534_2seg.png', 'val_535_2seg.png', 'val_535_3seg.png', 'val_535_4seg.png']
    tensor_label = []
    tensor_preds = []
    for label in labels:
        label_img = Image.open(os.path.join(label_path, label))
        label_tensor = to_tensor(label_img)
        tensor_label.append(label_tensor)
    for pred in preds:
        pred_img = Image.open(os.path.join(label_path, pred))
        pred_tensor = to_tensor(pred_img)
        '''plt.subplot(121)
        plt.imshow(pred_img)
        plt.subplot(122)
        plt.imshow(pred_tensor[0].cpu().numpy())
        plt.show()'''
        tensor_preds.append(pred_tensor)
    tensor_label = torch.stack(tensor_label).cuda()
    tensor_pred = torch.stack(tensor_preds).cuda()
    tensor_pred = (tensor_pred > 0.5).float()
    print('label shape:', tensor_label.shape, 'pred shape:', tensor_pred.shape)

    loss = SimpleFourierLoss(N=2).cuda()
    # loss = FourierLoss(N=2).cuda()
    weight = torch.tensor((0.5, 0.5), device='cuda')
    loss_val = loss(tensor_label.clone(), tensor_pred.clone(), weight)
    print(loss_val)
    bce = nn.BCELoss()
    bce_val = bce(tensor_pred.clone(), tensor_label.clone())
    print(bce_val)
    aff_loss = AffinityLoss(kernel_size=5).cuda()
    aff_loss_val = aff_loss(tensor_pred.clone(), tensor_label.clone())
    print(aff_loss_val)
    '''tensor_pred = (tensor_pred > 0.5).float()
    cc = kornia.contrib.connected_components(tensor_pred, num_iterations=500)
    cc2 = kornia.contrib.connected_components(tensor_label, num_iterations=500)
    for i in range(cc2.shape[0]):
        print('sample:', i,'  ', torch.unique(cc[i]))
        plt.subplot(121)
        plt.imshow(cc[i][0].cpu().numpy())
        plt.subplot(122)
        plt.imshow(cc2[i][0].cpu().numpy())
        plt.show()'''
