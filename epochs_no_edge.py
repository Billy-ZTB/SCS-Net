import sys
import torchvision
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter


class Epoch:
    def __init__(self, model,  loss, metrics, stage_name, use_distance=False,amp=False,update_freq=1,device="cuda", verbose=True):
        self.model = model
        self.update_freq = update_freq
        self.loss = loss
        self.amp = amp
        self.metrics = metrics
        self.metrics2 = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.use_distance= use_distance
        self._to_device()
        self.scaler = GradScaler(init_scale=2.0**14)    # 出现NaN的话调小

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        #print(logs)
        #str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        str_logs = ["{} - {:.4}".format(k, v.item() if isinstance(v, np.ndarray) else v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y,update=True):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            self.metrics.reset()
            self.metrics2.reset()
            for batch_index, data in enumerate(iterator):  # e: edge ground truth
                update=True
                if not self.use_distance:
                    x,y,edge =data
                    x, y, edge = \
                        x.to(self.device,non_blocking=True), y.to(self.device,non_blocking=True),edge.to(self.device,non_blocking=True)
                    gts = (y,edge)
                else:
                    x,y,edge,dist=data
                    x, y, edge, dist = x.to(self.device), y.to(self.device), edge.to(self.device), dist.to(self.device)
                    gts = (y,edge,dist)
                    if batch_index%self.update_freq!=(self.update_freq-1) and batch_index != len(dataloader) - 1:
                        update=False
                loss_value, y_pred = self.batch_update(x, gts, update)
                gt_data = y.cpu().detach().numpy().astype(int)
                pre_data = y_pred['seg_pred'][0].cpu().detach().numpy().astype(int)
                if np.isnan(pre_data).any():
                    print(f"Batch {batch_index} contains NaN values in pre_data. Skipping this batch.")
                    continue
                # update loss logs
                loss_value = loss_value.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'Loss': loss_meter.mean}
                logs.update(loss_logs)
                # update metrics logs
                gt_data[gt_data > 0.5] = 1
                gt_data[gt_data < 1] = 0
                pre_data[pre_data > 0.5] = 1
                pre_data[pre_data < 1] = 0
                self.metrics.add_batch(gt_data,pre_data)
                IoU = self.metrics.Intersection_over_Union()
                #IoU2 = self.metrics2.Intersection_over_Union()
                Pre = self.metrics.Precision()
                Recall = self.metrics.Recall()
                F1 = self.metrics.F1()
                acc = self.metrics.Pixel_Accuracy()
                #iou, miou, acc, recall, precision = self.metrics.value()
                iou_logs = {'iou': IoU[1],}
                            #'iou2': IoU2[1]}
                acc_logs = {'accuracy': acc}
                recall_logs = {'recall': Recall[1]}
                precision_logs = {'precision': Pre[1]}
                logs.update(iou_logs)
                logs.update(acc_logs)
                logs.update(recall_logs)
                logs.update(precision_logs)
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return logs, loss_value  # 本没有Loss_value，为了测试schedular.step添加


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, use_distance=False,amp=False,update_freq=1,device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            use_distance=use_distance,
            amp=amp,
            update_freq=update_freq,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()
        self.optimizer.zero_grad()

    def reinitalize_optimizer(self,optimizer):
        self.optimizer = optimizer

    def introduce_corner(self):
        self.loss.introduce_corner()

    def batch_update(self, x, y,update=True):
        #self.optimizer.zero_grad()
        self.model.cuda()
        if not self.amp:
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y) / self.update_freq
            loss.backward()
            if update:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            with autocast():
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
            self.scaler.scale(loss).backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)  # 梯度裁剪
            if update:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics,use_distance=False, device="cpu", amp=False, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            use_distance=use_distance,
            amp=amp,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, update=True):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss_dict = self.loss(prediction, y)
            #loss_dict = torch.tensor(.0,device='cuda')
        return loss_dict, prediction


'''try:
                # 检查是否包含 NaN 或 Inf
                if torch.isnan(prediction['seg_pred'][0]).any():
                    print("Error: NaN detected in y_pred['seg_pred'] at locations:")
                    print(torch.nonzero(torch.isnan(prediction['seg_pred'][0]), as_tuple=False))
                    raise ValueError("y_pred['seg_pred'] contains NaN values.")
                if torch.isinf(prediction['seg_pred'][0]).any():
                    print("Error: Inf detected in y_pred['seg_pred'] at locations:")
                    print(torch.nonzero(torch.isinf(prediction['seg_pred'][0]), as_tuple=False))
                    raise ValueError("y_pred['seg_pred'] contains Inf values.")

                np_pre_data = prediction['seg_pred'][0].cpu().detach().numpy()

                if np.isnan(np_pre_data).any():
                    print("Error: NaN detected in converted NumPy array at locations:")
                    print(np.argwhere(np.isnan(np_pre_data)))
                    raise ValueError("Converted NumPy array contains NaN values.")
                if np.isinf(np_pre_data).any():
                    print("Error: Inf detected in converted NumPy array at locations:")
                    print(np.argwhere(np.isinf(np_pre_data)))
                    raise ValueError("Converted NumPy array contains Inf values.")
                pre_data = np_pre_data.astype(int)
            except Exception as e:
                print(f"Exception caught: {e}")
                print("Debugging information:")
                print("Max value:", torch.max(prediction['seg_pred'][0]))
                print("Min value:", torch.min(prediction['seg_pred'][0]))
                print("First 10 values:", prediction['seg_pred'][0].flatten()[:10])  # 示例：打印前10个值
                print("Predicted shape:", prediction['seg_pred'][0].shape)
                exit(1)  # 退出程序以避免继续运行出错'''