import os
import random
import torch
import math
import numpy as np
import torch.nn.functional as F
from math import floor, sqrt
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomChoice
from torchvision.datasets.samplers import DistributedSampler
from timm.data import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100

# Visualizing Data Augmentation Reference: https://blog.joonas.io/193

_dataset_dict = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100
}

class TrainTransform:

    def __init__(self, resize, resize_mode, pad, scale, ratio, hflip, auto_aug, remode, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        # Horizontal Flip
        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        # Random Augmentation / Auto Augmentation
        if auto_aug:
            if auto_aug.startswith('ra'):
                transform_list.append(rand_augment_transform(auto_aug, {}))
            elif auto_aug.startswith('ta_wide'):
                transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_aug.startswith('aa'):
                policy = transforms.AutoAugmentPolicy('imagenet')
                transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))

        # Resize
        if resize_mode == 'RandomResizedCrop': # RandomCrop and then Resize
            transform_list.append(transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop': # Resize and RandomCrop
            transform_list.extend([
                transforms.Resize(resize, interpolation=interpolation),
                transforms.RandomCrop(resize, padding=pad)
            ])
        else:
            raise ValueError(f"{resize_mode} should be RandomResizedCrop and ResizedRandomCrop")

        # toTensor and Normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Random Erasing: this should be applied on Tensor (not PIL image)
        if remode:
            transform_list.append(RandomErasing(remode, max_count=3, device=None))

        self.transform_func = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_func(x)


class ValTransform:
    def __init__(self, size, resize_mode, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        # Make image bigger and then Crop
        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = [
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        self.transform_func = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_func(x)


class MixUp:
    def __init__(self, p=0.5, alpha=1.0, num_classes=1000):
        self.p = p
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype) # (B,) -> (B, num_classes)

        # Dirichlet distribution
        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        # re-order(shift) batches for MixUp two different images
        # e.g. batch[0] goes to batch[1] ...
        batch_roll = batch.roll(shifts=1, dims=0) # (B, C, H, W)
        target_roll = target.roll(shifts=1, dims=0) # (B, num_classes)

        # MixUp
        batch = batch * (1-ratio) + batch_roll * ratio
        target = target * (1-ratio) + target_roll * ratio

        return batch, target


class CutMix:
    def __init__(self, p=0.5, alpha=1.0, num_classes=1000):
        self.p = p
        self.alpha = alpha
        self.num_classes = num_classes

    @torch.inference_mode()
    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        B, C, H, W = batch.shape
        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        # re-order(shift) batches for CutMix two different images
        # e.g. batch[0] goes to batch[1] ...
        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        height_half = int(0.5 * sqrt(ratio) * H)
        width_half = int(0.5 * sqrt(ratio) * W)
        r = int(random.random() * H)
        c = int(random.random() * W)

        start_x = max(r - height_half, 0)
        end_x = min(r + height_half, H)
        start_y = max(c - width_half, 0)
        end_y = min(c + width_half, W)

        ratio = 1 - ((end_x - start_x) * (end_y - start_y) / (H * W))

        # CutMix two images
        batch[:, :, start_x:end_x, start_y:end_y] = batch_roll[:, :, start_x:end_x, start_y:end_y]
        target = target * (1-ratio) + target_roll * ratio

        return batch, target


def get_dataset(args):
    dataset_name = _dataset_dict[args.dataset_name]
    train_transform = TrainTransform(args.train_size, args.train_resize_mode, args.random_crop_pad, args.random_crop_scale, args.random_crop_ratio,\
                                     args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std)
    val_transform = ValTransform(args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation, args.mean, args.std)

    # e.g. ImageNet
    if args.dataset_name == 'ImageFolder':
        train_dataset = dataset_name(os.path.join(args.data_dir, args.train_split), transform=train_transform)
        val_dataset = dataset_name(os.path.join(args.data_dir, args.val_split), transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    # e.g. CIFAR10/100
    elif args.dataset_name in _dataset_dict.keys():
        train_dataset = dataset_name(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = dataset_name(root=args.data_dir, train=False, download=True, transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    else:
        raise ValueError(f'Dataset {args.dataset_name} not supported yet... Try make a code for it!')

    return train_dataset, val_dataset


def get_dataloader(train_dataset, val_dataset, args):

    # Create Sampler: defines the strategy to draw samples from the dataset
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset) # it is equal as DataLoader(shuffle=True)
        val_sampler = SequentialSampler(val_dataset) # it is equal as DataLoader(shuffle=False)

    # Create collate_fn for MixUp & CutMix (collate_fn: merges a list of samples to form a mini-batch of Tensors)
    mix_collate = []
    if args.mixup:
        mix_collate.append(MixUp(alpha=args.mixup, num_classes=args.num_classes))
    if args.cutmix:
        mix_collate.append(CutMix(alpha=args.cutmix, num_classes=args.num_classes))

    if mix_collate:
        mix_collate = RandomChoice(mix_collate) # choose one: MixUp or CutMix
        collate_fn = lambda batch: mix_collate(*default_collate(batch))
    else:
        collate_fn = None # if None, will use default_collate

    # Create DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,\
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory, worker_init_fn=args.seed_worker)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,\
                                num_workers=args.num_workers, collate_fn=None, pin_memory=args.pin_memory, worker_init_fn=args.seed_worker)

    # 'real_batch_size'
    args.total_batch_size = args.batch_size * args.grad_accum_step

    # iters_per_epoch: 'real_batch_size' iterations per epoch
    args.iters_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_step)

    return train_dataloader, val_dataloader

