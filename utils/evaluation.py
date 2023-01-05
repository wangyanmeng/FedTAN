# Performance evaluation

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def Inference_loss(device, val_loader, criterion, model, half_flag):
    """
    Run evaluation
    """
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            if half_flag:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure record loss
            losses.update(loss.item(), input.size(0))

    return losses.avg


def Inference_accuracy(device, val_loader, model, half_flag):
    """
    Run evaluation
    """
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)

            if half_flag:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            output = output.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

    return top1.avg