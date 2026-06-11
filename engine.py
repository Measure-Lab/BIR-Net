"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, mix=False
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, mix)
            if isinstance(outputs, list):
                if isinstance(outputs[0], list):
                    loss = criterion(outputs[0][0], outputs[0][1], outputs[1][0], outputs[1][1], targets)
                    # loss_trans = criterion(, targets)
                    # loss = loss_conv + loss_trans
                else:
                    loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                    loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list) and not mix:
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    total_probs = []
    total_targets = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        total_targets.extend(target.detach().cpu().numpy().tolist())
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            acc1_head1 = accuracy(output[0], target, topk=(1,))[0][0]
            acc1_head2 = accuracy(output[1], target, topk=(1,))[0][0]
            acc1_total_list, probs = accuracy(output[0] + output[1], target, topk=(1,))
            acc1_total = acc1_total_list[0]
            if isinstance(total_probs, list):
                total_probs = probs
            else:
                total_probs = np.concatenate([total_probs, probs], axis=0)
            acc5_total = accuracy(output[0] + output[1], target, topk=(5,))[0][0]
        else:
            # others
            acc1, acc5, _ = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    n_classes = max(total_targets) + 1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(total_targets, total_probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    import matplotlib.pyplot as plt
    from itertools import cycle
    from matplotlib.ticker import FuncFormatter
    lw = 2
    # Plot all ROC curves
    plt.figure()
    labels = ['Category 0','Category 1','Category 2','Category 3','Category 4']
    plt.plot(fpr["macro"], tpr["macro"], color='darkorange', lw=lw, label='macro-average ROC curve (area = {0:0.4f})'.format(roc_auc["macro"]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity (%)')
    plt.ylabel('Sensitivity (%)')
    plt.title('ROC Curve of CUB200')
    def to_percent(temp, position):
        return '%1.0f'%(100*temp)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.legend(loc="lower right")
    plt.savefig('./auc-roc.png')
    # plt.show()


    # 输出并返回
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@heads_top5 {heads_top5.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, heads_top5=metric_logger.acc5, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
