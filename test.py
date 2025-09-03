import fileinput
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from Dataset import *
from Dataset.Buildings_dataset import BuildingDataset
from Model.UANet import *
from Model.DoubleBranch import DualBranchNet
from Model.DoubleBranchAblation import DualBranchNetAblation, SingleBranch
from Model.DualBranchAblation2 import DualBranchNetAblation2
from metric.metrics import *
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Test the model')
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--dataset', type=str, default='NewInria')
parser.add_argument('--baseline',action='store_true',help='whether this is the baseline')
parser.add_argument('--ablation', action='store_true', help='Whether the model is ablation model')
parser.add_argument('--backbone', type=str, default='vgg16', help='The backbone of the model')
parser.add_argument('--glag', action='store_true', help='Whether use global aggregation')
parser.add_argument('--pdc', action='store_true', help='Whether use pyramid dilated convolution')
parser.add_argument('--fusion', action='store_true', help='Whether use fusion module')
parser.add_argument('--mcb', action='store_true', help='Whether use multi-scale context block')
parser.add_argument('--dilation_ratio', type=float, default=0.02, help='Dilation ratio for boundary extraction')
args = parser.parse_args()


def make_boundary(mask, dilation_ratio=0.02):
    mask = mask.squeeze().astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    erosion = cv2.erode(new_mask, np.ones((3, 3), np.uint8), iterations=dilation)
    mask_erosion = erosion[1:-1, 1:-1]
    boundary = mask - mask_erosion
    return boundary

def load_best_model(model, path):
    num = []
    for pth in os.listdir(os.path.join(path,'weight')):
        if 'Best' in pth:
            number = int(pth[pth.find('epoch')+len('epoch'):pth.rfind('.pth')])
            num.append(number)
    best = max(num)
    print('Best model is in epoch',best)
    model.load_state_dict(torch.load(os.path.join(path,'weight',f'Best_model_epoch{best}.pth')))
    return model

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = Evaluator(2)
    b_metrics = Evaluator(2)
    dataset = BuildingDataset(split='test', dataset_name=args.dataset)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = DualBranchNetAblation(backbone=args.backbone,
                                  glag=args.glag,
                                  pdc=args.pdc,
                                  fusion=args.fusion,
                                  mcb=args.mcb
                                  ).to(DEVICE) \
         if not args.baseline else SingleBranch(backbone=args.backbone).to(DEVICE)
    #model = UANet_VGG(channel=32, num_classes=2).cuda()
    #state_dict = torch.load(r'D:\ztb\DeepLearning\Uncertainty-aware-Network-master\model_weights\CrowdAI\UANet_VGG\UANet_VGG-v4.ckpt',map_location=DEVICE)['state_dict']

    '''for key in list(state_dict.keys()):
        state_dict[key.replace('net.','')] = state_dict.pop(key)
        #print(key.replace('net.',''))
    model.load_state_dict(state_dict)'''
    model = load_best_model(model, args.output)
    model.eval()

    with tqdm(test_loader, total=len(test_loader)) as pbar:
        for idx, (image, label, name) in enumerate(pbar):
            input = image.to(DEVICE)
            gt = label.clone()
            size = gt.size()[2:]
            with torch.no_grad():
                pred_dict = model(input)
            gt = (gt.cpu().detach().numpy()>0.5).astype(np.uint8)
            pred = pred_dict['seg_pred'][0]

            '''if pred.size()[1]>1:
                pred = nn.Softmax(dim=1)(pred).argmax(dim=1,keepdim=True)'''
                #pred = pred[:,1,:,:].unsqueeze(1)
            pred = F.interpolate(pred, size, mode='bilinear', align_corners=True).cpu().detach().numpy().astype(int)

            gt[gt > 0.5] = 1
            gt[gt < 1] = 0
            pred[pred > 0.5] = 1
            pred[pred < 1] = 0

            '''edge = pred_dict['edge_pred'][0]
            plt.subplot(1,2,1)
            plt.imshow(np.squeeze(pred))
            plt.subplot(1,2,2)
            plt.imshow(torch.sigmoid(edge).squeeze().cpu().detach().numpy())
            plt.show()'''

            gt_bound = make_boundary(gt,dilation_ratio=args.dilation_ratio)
            pred_bound = make_boundary(pred,dilation_ratio=args.dilation_ratio)
            metrics.add_batch(gt, pred)
            b_metrics.add_batch(gt_bound, pred_bound)

        IoU = metrics.Intersection_over_Union()
        Pre = metrics.Precision()
        Recall = metrics.Recall()
        F1 = metrics.F1()
        acc = metrics.Pixel_Accuracy()
        BIoU = b_metrics.Intersection_over_Union()
        BF1 = b_metrics.F1()
        print('iou:', IoU[1], '\naccuracy:', acc, '\nprecision:', Pre[1], '\nrecall:', Recall[1], '\nF1:', F1[1],
              '\nBoundary IoU:', BIoU[1], '\nBoundary F1:', BF1[1])

    with open(os.path.join(args.output, 'readme.txt'), 'a') as f:
        f.write(
            f'iou:{IoU[1]}\naccuracy:{acc}\nprecision:{Pre[1]}\nrecall:{Recall[1]}\nF1:{F1[1]}\nBoundary IoU:{BIoU[1]}\nBoundary F1:{BF1[1]}')
