import os
from email.policy import strict
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.autograd.profiler as profiler
from sympy import false
from torch.autograd import detect_anomaly

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Dataset.Buildings_dataset import BuildingDataset
#from Model.RCMEdgeUnet_S import *
#from Model.DoubleBranch import DualBranchNet
#from Model.DualBranchAblation2 import DualBranchNetAblation2
from Model.DoubleBranchAblation import DualBranchNetAblation, SingleBranch
#from Model.RCMedgeUnet import *
from Utils.epochs_no_edge import *
#from metric.iou_single_class import IoU_Foreground
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from Losses.loss import *
from metric.metrics import *
from torch.cuda.amp import GradScaler, autocast


parser = argparse.ArgumentParser(description='DualBranchNet Ablation')
parser.add_argument('--dataset',type=str, default='CTC', help='dataset name')
parser.add_argument('--output',type=str, default='DualBranch', help='output folder name')
parser.add_argument('--backbone',type=str, default='vgg16', help='backbone')
parser.add_argument('--batch_size',type=int, default=8, help='batch size')
parser.add_argument('--update_interval',type=int, default=1, help='update interval')
parser.add_argument('--amp',action='store_true', help='use amp (default: False)')
parser.add_argument('--baseline',action='store_true',help='whether this is the baseline')
parser.add_argument('--glag', action='store_true', help='use glag (default: False)')
parser.add_argument('--pdc', action='store_true', help='use pdc (default: False)')
parser.add_argument('--fusion', action='store_true', help='use fusion (default: False)')
parser.add_argument('--mcb', action='store_true', help='use mcb (default: False)')
parser.add_argument('--edge_weight',type=float, default=10, help='edge loss weight')
parser.add_argument('--T_0',type=int, default=7, help='T_0 for CosineAnnealingWarmRestarts')
parser.add_argument('--T_mult',type=int, default=2, help='T_mult for CosineAnnealingWarmRestarts')
parser.add_argument('--lr',type=float, default=3e-4, help='initial learning rate')
parser.add_argument('--detect_anomaly',action='store_true', help='detect anomaly (default: False)')
parser.add_argument('--remove_iou_loss','-remove_iou',action='store_false', help='add this if you want to remove iou loss (default: True)')
parser.add_argument('--remove_deep_supervision','-remove_ds',action='store_false', help='add this if you want to remove deep supervision (default: True)')
args = parser.parse_args()

train_batch_size = args.batch_size//args.update_interval
amp = args.amp #if dataset_name=='CrowdAI' else False
use_distance = False
epoch_to_add_corner = -1  # 不加入cornerpointloss就设负值
affinity = False
point_sprvsn = False
detect_anomaly_ = False
consistency = 0
edge_weight = args.edge_weight
#print('args.edge_weight:',args.edge_weight)
#T_max = args.T_0 * (1+args.T_mult)
T_max = 50
root = os.path.join('C:\ZTB\Results', args.dataset, f'{args.backbone.upper()}_{args.output}')
exp_name = args.backbone
writer = SummaryWriter(os.path.join(root, 'exp', exp_name))


def draw_plot(step, var, value, epoch):
    plt.figure()
    plt.plot(range(0, len(value)), value, marker='o', linestyle='-')
    plt.title(f'{step}ing {var} vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'{var}')
    plt.legend([f'{step}ing {var}'])
    if not os.path.isdir(os.path.join(root, f'{step}ing_{var}')):
        os.makedirs(os.path.join(root, f'{step}ing_{var}'))
    plt.savefig(os.path.join(root, f'{step}ing_{var}', f'{step}ing_{var}_epoch{epoch}.png'))


def reinitialize_optimizer(optimizer, loss, lr=6e-4):
    for param_group in optimizer.param_groups:
        if 'params' in param_group:
            for p in param_group['params']:
                if any(torch.equal(p, lp) for lp in loss.parameters()):
                    print('Found loss parameter:', p)
                    param_group['lr'] = lr
                    print('Reinitialize lr for loss parameters')


if __name__ == '__main__':
    import time

    print('Training started at ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    torch.cuda.empty_cache()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    #model = VGGUNet().to(DEVICE)
    #model = DualBranchNet(backbone=backbone).to(DEVICE)
    if args.baseline:
        model = SingleBranch(backbone=args.backbone).to(DEVICE)
    else:
        model = DualBranchNetAblation(backbone=args.backbone,
                                  glag=args.glag,
                                  pdc=args.pdc,
                                  mcb=args.mcb,
                                  fusion=args.fusion).to(DEVICE)

    def hook(module, input, output):
        # print('hook of ', module)
        if isinstance(output, tuple):
            for idx, o in enumerate(output):
                if isinstance(o, list):
                    for i, oo in enumerate(o):
                        if torch.isnan(oo).any():
                            print(
                                f"NaN detected in {module},No{idx} of this tuple, No{i} of this list, output shape: {oo.shape}")
                        if torch.isinf(oo).any():
                            print(
                                f"Inf detected in {module},No{idx} of this tuple, No{i} of this list, output shape: {oo.shape}")
                elif torch.isnan(o).any():
                    print(f"NaN detected in {module}, No{idx} of this tuple, output shape: {o.shape}")
                elif torch.isinf(o).any():
                    print(f"Inf detected in {module}, No{idx} of this tuple, output shape: {o.shape}")
        elif isinstance(output, list):
            for i, o in enumerate(output):
                if torch.isnan(o).any():
                    print(f"NaN detected in {module}, No{i} output shape: {o.shape}")
                if torch.isinf(o).any():
                    print(f"Inf detected in {module}, No{i} output shape: {o.shape}")
        elif torch.isnan(output).any():
            print(f"NaN detected in {module}, output shape: {output.shape}")
        elif torch.isinf(output).any():
            print(f"Inf detected in {module}, output shape: {output.shape}")


    # 添加 Hook
    # for name, module in model.named_modules():
    #    print(name)
    #    module.register_forward_hook(hook)

    train_dataset = BuildingDataset('train', args.dataset)
    valid_dataset = BuildingDataset('valid', args.dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    loss = DeepSupervisionCELoss(edge_weight=args.edge_weight,use_iou=args.remove_iou_loss,deep_supervision=args.remove_deep_supervision)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)  # 5e-4?
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=5e-5)   # eta_min 5e-5

    latest_epoch = -1
    if not os.path.isdir(os.path.join(root, 'weight')):
        os.makedirs(os.path.join(root, 'weight'))
    pth_files = os.listdir(os.path.join(root, 'weight'))

    if len(pth_files) > 0:
        epoch_num = list()
        for pth in pth_files:
            if pth.startswith('.') or 'epoch' not in pth:
                continue
            if pth.startswith('Best'):
                num = int(pth[pth.find('epoch') + len('epoch'):pth.rfind('.')])  # find从左到右找，rfind从右到左
                epoch_num.append(num)
        latest_epoch = max(epoch_num)
        state_dict = [f for f in pth_files if str(latest_epoch) in f][0]
        state_dict = torch.load(os.path.join(root, 'weight', state_dict))
        #filtered_dict = {k: v for k, v in state_dict.items() if 'reclassify' not in k}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(
            torch.load(os.path.join(root, 'weight&optimizer', f'Optimizer_epoch{latest_epoch}.pth')))
        scheduler.load_state_dict(
            torch.load(os.path.join(root, 'weight&optimizer', f'Scheduler_epoch{latest_epoch}.pth')))
        print(f'Last BEST epoch {latest_epoch} model loaded!\n')

    # loss = JointDecoupleCascadeLoss() #nn.BCEWithLogitsLoss() #DeepSupervisionCELoss()
    metrics = Evaluator(2)

    train_epoch = TrainEpoch(
        model,
        update_freq=args.update_interval,
        use_distance=use_distance,
        amp=amp,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        use_distance=use_distance,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    train_loss, train_iou, valid_loss, valid_iou = [], [], [], []
    weight_save_dir = os.path.join(root, 'weight')
    opt_sche_save_dir = os.path.join(root, 'weight&optimizer')
    if not os.path.isdir(weight_save_dir):
        os.makedirs(weight_save_dir)
    if not os.path.isdir(opt_sche_save_dir):
        os.makedirs(opt_sche_save_dir)

    if latest_epoch >= 0:  # 如果不是从头训练，valid最好epoch的iou作为初始最大值
        logs, _ = valid_epoch.run(valid_loader)
        max_score = logs['iou']
        print('\nTraining from epoch ', latest_epoch)
    else:
        print('\nTraining from scratch.')

    no_optim_epoch = 0
    best_epoch = 0

    torch.autograd.set_detect_anomaly(detect_anomaly_)

    for i in range(latest_epoch + 1, T_max):
        print('\nEpoch: {}'.format(i))
        if 0 <= epoch_to_add_corner <= i:
            train_epoch.introduce_corner()
        train_logs, tloss = train_epoch.run(train_loader)
        if np.isscalar(train_logs['Loss']):
            train_l_now = [train_logs['Loss']]
        else:
            train_l_now = [v if np.isscalar(v) else v[0] for v in train_logs['Loss']]
        train_loss.append(train_l_now)
        # train_loss.append(train_logs['Loss'])
        train_iou.append(train_logs['iou'])
        valid_logs, vloss = valid_epoch.run(valid_loader)
        if np.isscalar(valid_logs['Loss']):
            valid_l_now = [valid_logs['Loss']]
        else:
            valid_l_now = [v if np.isscalar(v) else v[0] for v in valid_logs['Loss']]
        valid_loss.append(valid_l_now)
        valid_iou.append(valid_logs['iou'])
        scheduler.step()

        with open(os.path.join(root, 'output_per_epoch.txt'), 'a') as f:
            f.write(
                f"Epoch {i}:\nTrain: Loss:{train_logs['Loss'].item() if isinstance(train_logs['Loss'], np.ndarray) else train_logs['Loss']:.4f},IoU:{train_logs['iou']:.4f},Acc:{train_logs['accuracy']:.4f},"
                f"Recall:{train_logs['recall']:.4f},Precision:{train_logs['precision']:.4f}\n"
                f"Valid: Loss:{valid_logs['Loss'].item() if isinstance(valid_logs['Loss'], np.ndarray) else valid_logs['Loss']:.4f},IoU:{valid_logs['iou']:.4f},Acc:{valid_logs['accuracy']:.4f},"
                f"Recall:{valid_logs['recall']:.4f},Precision:{valid_logs['precision']:.4f}\n")

        if max_score <= valid_logs['iou']:
            max_score = valid_logs['iou']
            no_optim_epoch = 0
            best_epoch = i
            torch.save(model.state_dict(), os.path.join(weight_save_dir, f'Best_model_epoch{i}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(opt_sche_save_dir, f'Optimizer_epoch{i}.pth'))
            torch.save(scheduler.state_dict(), os.path.join(opt_sche_save_dir, f'Scheduler_epoch{i}.pth'))
            print(f'Best Model {i} saved!')
        else:
            torch.save(model.state_dict(), os.path.join(weight_save_dir, f'Weight_epoch{i}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(opt_sche_save_dir, f'Optimizer_epoch{i}.pth'))
            if hasattr(loss, 'parameters') and list(loss.parameters()):
                torch.save(loss.state_dict(), os.path.join(opt_sche_save_dir, f'Loss_epoch{i}.pth'))
            torch.save(scheduler.state_dict(), os.path.join(opt_sche_save_dir, f'Scheduler_epoch{i}.pth'))
            no_optim_epoch += 1
            print(f'Epoch {i} Model saved!')
        '''if no_optim_epoch > 3:
            model.load_state_dict(torch.load(os.path.join(weight_save_dir, f'Best_model_epoch{best_epoch}.pth')))
            optimizer.load_state_dict(torch.load(os.path.join(opt_sche_save_dir, f'Optimizer_epoch{best_epoch}.pth')))
            scheduler.step()
            no_optim_epoch=0 # 重跑
            print(f'Reload epoch{best_epoch} state dict.')
            print('Reduce learning rate to ',optimizer.state_dict()['param_groups'][0]['lr'])'''
        if (i % 10 == 0) and (i > 0):
            draw_plot('train', 'loss', train_loss, i)
            draw_plot('train', 'iou', train_iou, i)
            draw_plot('valid', 'loss', valid_loss, i)
            draw_plot('valid', 'iou', valid_iou, i)
