
# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --standalone train.py resnet34 --use_wandb --cutmix

import os
import shutil
import argparse
import torch
import time
import wandb
import torch.nn as nn
import torch.optim as optim
from setup import init_wandb, init_distributed_mode
from utils import Metric, count_params
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import v2
from models.resnet_for_cifar import resnet34, resnet50
from models.seresnet_for_cifar import seresnet34, seresnet50
from models.pyrmdnet_for_cifar import pyramidnet110_48, se_pyramidnet110_48, pyramidnet164_270, se_pyramidnet164_270
from models.gate_resnet_for_cifar import gate_resnet34
from models.gate_seresnet_for_cifar import gate_seresnet34
from models.experiment_model import fusion_resnet34


parser = argparse.ArgumentParser(description='Some Hyper-parameters to Train Deep Learning Model')

parser.add_argument('--proj', type=str, default='ResNet Experiments', help='Project Name')
parser.add_argument('--expname', type=str, default='test', help='Experiment name')
parser.add_argument('model', type=str, help='Model Options: (resnet, pyramidnet)')
parser.add_argument('--dataset', dest='dataset', type=str, default='cifar100', help='Dataset Options: (cifar10, cifar100)')
parser.add_argument('--bs', dest='batch_size',  metavar='', type=int, default=128, help='Batch Size')
parser.add_argument('--optimizer', dest='optimizer', metavar='', type=str, default='SGD', help='Optimizer')
parser.add_argument('--lossfn', dest='loss_func', metavar='', type=str, default='CrossEntropy', help='Loss Function')
parser.add_argument('--lr', '--learning_rate', metavar='', type=float, default=0.1, help='Learning Rate')
parser.add_argument('--momentum', dest='momentum', metavar='', type=float, default=0.9, help='Momentum')
parser.add_argument('--wd', '--weight_decay', metavar='', type=float, default=1e-4, help='Weight Decay')
parser.add_argument('--epochs', dest='epochs',  metavar='', type=int, default=100, help='Number of Epochs')
parser.add_argument('--cutmix', dest='cutmix', action='store_true', default=False, help='Use CutMix or MixUp')
parser.add_argument('--print_frequency', dest='print_frequency', metavar='', type=int, default=50, help='Print Frequency')
parser.add_argument('--verbose', dest='verbose', action='store_true', default=True, help='Verbose')
parser.add_argument('--use_wandb', dest='wandb', action='store_true', help='Use Wandb')

parser.add_argument('--cuda', dest='cuda', type=str, default='', help='GPU Device ID')


best_acc1 = 0
best_acc5 = 0

def main():

    global args, best_acc1, best_acc5

    args = parser.parse_args()

    # Initial Setup
    init_distributed_mode(args)
    init_wandb(args)

    # Data Preprocessing for CIFAR dataset
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        dset_train = CIFAR10(root="./data", download=True, train=True, transform=transform_train)
        dset_val = CIFAR10(root="./data", download=True, train=False, transform=transform_train)

        train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=2)

        num_classes = 10

    elif args.dataset == 'cifar100':

        dset_train = CIFAR100(root="./data", download=True, train=True, transform=transform_train)
        dset_val = CIFAR100(root="./data", download=True, train=False, transform=transform_train)

        if args.distributed:

            train_sampler = DistributedSampler(dataset=dset_train, shuffle=True)
            val_sampler = DistributedSampler(dataset=dset_val, shuffle=False)

            train_loader = DataLoader(
                dset_train,
                batch_size=int(args.batch_size / args.world_size),
                shuffle=False,
                num_workers=4,
                sampler=train_sampler,
                pin_memory=True
            )
            val_loader = DataLoader(
                dset_val,
                batch_size=int(args.batch_size / args.world_size),
                shuffle=False,
                num_workers=4,
                sampler=val_sampler,
                pin_memory=True
            )
        else:
            train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=2)

        num_classes = 100

    else:
        raise Exception(f'Unknown Dataset: {args.dataset}')

    # Model configuration (ResNet, SEResNet, PyramidNet)
    if args.model == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif args.model == 'seresnet34':
        model = seresnet34(num_classes=num_classes)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=num_classes)
    elif args.model == 'seresnet50':
        model = seresnet50(num_classes=num_classes)
    elif args.model == 'pyramidnet110_48':
        model = pyramidnet110_48(num_classes=num_classes)
    elif args.model == 'se_pyramidnet110_48':
        model = se_pyramidnet110_48(num_classes=num_classes)
    elif args.model == 'pyramidnet164_270':
        model = pyramidnet164_270(num_classes=num_classes)
    elif args.model == 'se_pyramidnet164_270':
        model = se_pyramidnet164_270(num_classes=num_classes)
    elif args.model == 'gate_resnet34':
        model = gate_resnet34(num_classes=num_classes)
    elif args.model == 'gate_seresnet34':
        model = gate_seresnet34(num_classes=num_classes)
    elif args.model == 'fusion_resnet34':
        model = fusion_resnet34(num_classes=num_classes)
    else:
        raise Exception(f'Unknown Model Name: {args.model}')

    model = model.to(args.gpu)

    if args.distributed:
        # To get same calculation results as single GPU, should synchronize b/t GPUs when doing BatchNorm
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Warped by DDP (Distributed Data Parallel)
        model = DDP(module=model, device_ids=[args.gpu])

    # Loss function (default: CrossEntropyLoss)
    loss_func = nn.CrossEntropyLoss()

    # Optimizer (default: SGD)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    # Learning Rate Scheduler (default: cosine)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Model info.
    params = count_params(model)
    print(f"Training Started (Model: {args.model}, # of Parameters: {params})")
    print(model)

    for epoch in range(args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        learning_rate_scheduler(optimizer, epoch)

        # train for one epoch
        train_top1_acc, train_top5_acc, train_loss = train(model, train_loader, optimizer, loss_func, epoch, args.cutmix)

        # val for one epoch
        val_top1_acc, val_top5_acc, val_loss = validation(model, val_loader, loss_func, epoch)


        # fixme : try to plot errors per iter.
        # 🐝 plot in wandb.ai (train loss per epoch)
        if args.wandb and args.is_rank_zero:
            wandb.log({'train_loss': train_loss,
                       'top-1 train accuracy': train_top1_acc,
                       'top-5 train accuracy': train_top5_acc,
                       'top-1 val accuracy': val_top1_acc,
                       'top-5 val accuracy': val_top5_acc
                       })

        is_best = (val_top1_acc >= best_acc1)
        best_acc1 = max(val_top1_acc, best_acc1)
        if is_best:
            best_acc5 = val_top5_acc

        print('\n')
        print(f'*** Current Best (val) Accuracy (top-1 acc: {val_top1_acc:.4f}, top-5 acc: {val_top5_acc:.4f}) ***')
        print('\n')

        save_checkpoint({
            'experiment': args.expname,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_acc5': best_acc5,
            'optimizer': optimizer.state_dict()
        }, is_best)

    print(f'Best (val) Accuracy (top-1 acc: {best_acc1}, top-5 acc: {best_acc5})')

    # 🐝 finish wandb.ai
    if args.wandb and args.is_rank_zero:
        wandb.finish(quiet=True)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = f"{args.proj}_checkpoint/{args.expname}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

    # copy best state file
    if is_best:
        shutil.copyfile(filename, f'{args.proj}_checkpoint/{args.expname}/' + 'model_best.pth.tar')


def train(
        model,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        # lr_scheduler: torch.optim.lr_scheduler,
        loss_func: torch.nn,
        epoch: int,
        do_mixup_or_cutmix: bool=False,
):
    losses = Metric(reduce=True)
    top1_accs = Metric(reduce=True)
    top5_accs = Metric(reduce=True)
    batch_time = Metric()
    data_time = Metric()

    # Data Augmentation: CutMix or MixUp
    if do_mixup_or_cutmix is True:
        cutmix = v2.CutMix(num_classes=100)
        mixup = v2.MixUp(num_classes=100)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


    end = time.time()
    current_lr = get_learning_rate(optimizer)[0]

    # switch model to training mode
    model.train()
    for idx, it in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_batch, y_batch = it

        # Models and Data should be in the same device
        # memory format: (B, C, H, W) -> (B, H, W, C)
        # still same shape but different stride: (C*H*W, H*W, W, 1) -> (C*H*W, 1, W*C, C)
        model = model.to(device=args.device, memory_format=torch.channels_last)
        x_batch = x_batch.to(device=args.device, memory_format=torch.channels_last)
        y_batch = y_batch.to(device=args.device)

        if do_mixup_or_cutmix is True:
            # CutMix or MixUp input batches
            x_batch, y_batch = cutmix_or_mixup(x_batch, y_batch)

            target_a_idx = y_batch.topk(k=2, dim=-1).indices[:, 0]
            target_b_idx = y_batch.topk(k=2, dim=-1).indices[:, 1]
            lam = y_batch.topk(k=2, dim=-1).values[:, 0][0]

            # Forwarding and compute loss
            output = model(x_batch)
            loss = loss_func(output, target_a_idx) * lam + loss_func(output, target_b_idx) * (1 - lam)

            y_batch = target_a_idx

        else:
            # Forwarding and compute loss
            output = model(x_batch)
            loss = loss_func(output, y_batch)

        top1_acc, top5_acc = accuracy(output, y_batch, topk=(1,5))

        losses.update(loss, x_batch.size(0))
        top1_accs.update(top1_acc, x_batch.size(0))
        top5_accs.update(top5_acc, x_batch.size(0))

        # fixme : add AMP to speed up!
        # This will zero out the gradients for this minibatch.
        optimizer.zero_grad()

        loss.backward() # Backpropagation
        optimizer.step() # Perform parameter updates
        # lr_scheduler.step() # Learning Rate updates

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_frequency == 0 and args.verbose is True:
            print(
                f'Epoch: [{epoch}/{args.epochs}][{idx}/{len(train_dataloader)}]\t\
                LR: {current_lr:.6f}\t\
                Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t\
                Data {data_time.value:.3f} ({data_time.avg:.3f})\t\
                Loss {losses.value.item():.4f} ({losses.avg.item():.4f})\t\
                Top-1 acc {top1_accs.value.item():.4f} ({top1_accs.avg.item():.4f})\t\
                Top-5 acc {top5_accs.value.item():.4f} ({top5_accs.avg.item():.4f})\t'
            )

    return top1_accs.avg.item(), top5_accs.avg.item(), losses.avg.item()



def validation(model, val_dataloader, loss_func, epoch):

    losses = Metric(reduce=True)
    top1_accs = Metric(reduce=True)
    top5_accs = Metric(reduce=True)
    batch_time = Metric()

    # switch model to evaluate mode
    model.eval()

    end = time.time()
    print('\n')
    for idx, it in enumerate(val_dataloader):

        x_batch, y_batch = it
        # memory format: (B, C, H, W) -> (B, H, W, C)
        model = model.to(device=args.device, memory_format=torch.channels_last)
        x_batch = x_batch.to(device=args.device, memory_format=torch.channels_last)
        y_batch = y_batch.to(device=args.device)

        output = model(x_batch)
        loss = loss_func(output, y_batch)

        top1_acc, top5_acc = accuracy(output, y_batch, topk=(1,5))

        losses.update(loss, x_batch.size(0))
        top1_accs.update(top1_acc, x_batch.size(0))
        top5_accs.update(top5_acc, x_batch.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_frequency == 0 and args.verbose == True:
            print(f'Test (on val set): [{epoch}/{args.epochs}][{idx}/{len(val_dataloader)}]\t\
                  Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t\
                  Top-1 acc {top1_accs.value.item():.4f} ({top1_accs.avg.item():.4f})\t\
                  Top-5 acc {top5_accs.value.item():.4f} ({top5_accs.avg.item():.4f})\t'
            )


    return top1_accs.avg.item(), top5_accs.avg.item(), losses.avg.item()



def learning_rate_scheduler(optimizer, epoch):
    """
    Decaying the learning rate for CIFAR dataset
    """
    new_lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_learning_rate(optimizer):
    """
    Return learning rate history
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # if top-5: (B, 5)
    # transpose to (5, B) for easily indexing
    # .contiguous() makes a new transposed tensor(= new stride) without referring to the original tensor
    # so use .contiguous() after .permute() or .transpose()
    pred = pred.t().contiguous()
    correct = pred.eq(target.reshape(1, -1).contiguous().expand_as(pred)) # (5, B)

    acc = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).contiguous().float().sum(dim=0, keepdim=True)
        acc.append(correct_k.mul_(100.0 / batch_size))

    return acc



if __name__ == '__main__':
    main()




