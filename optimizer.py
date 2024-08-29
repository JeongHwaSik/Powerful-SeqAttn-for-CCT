import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, StepLR, ExponentialLR,\
    LambdaLR, SequentialLR, OneCycleLR

class BinaryCrossEntropy(nn.Module):
    """
    Binary Cross Entropy (timm)
    :arg
        label_smoothing: multi-class loss in bce.
        bce_target: remove uncertain target used with cutmix.
    """
    def __init__(self, label_smoothing=0.1, bce_target=None):
        super(BinaryCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.bce_target = bce_target

    def forward(self, x, y):
        if x.shape != y.shape: # if MixUp or CutMix is not used, make hard-label: (B,) into smooth-label: (B, num_classes)
            smooth = self.label_smoothing / x.size(-1)
            label = 1.0 - self.label_smoothing + smooth
            smooth_target = torch.full_like(x, smooth) # (B, num_classes)
            y = torch.scatter(input=smooth_target, dim=-1, index=y.long().view(-1, 1), value=label)

        # fixme: why need this condition???
        if self.bce_target:
            y = torch.gt(y, self.bce_target).long()
        return F.binary_cross_entropy_with_logits(x, y, reduction='mean')


class NativeScalerWithGradAccum:
    def __init__(self):
        """
        NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer) # unscaling gradients b/f gradient clipping
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm) # gradient clipping for stable update (threshold = grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



def get_optimizer_and_scheduler(model, args):
    parameter = model.parameters()
    total_iter = args.epochs * args.iters_per_epoch
    warmup_iter = args.warmup_epochs * total_iter

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = SGD(params=parameter, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(params=parameter, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(params=parameter, lr=args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"{args.optimizer} not implemented yet")

    # learning rate scheduler (per iter.)
    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=total_iter-warmup_iter, eta_min=args.min_lr)
    elif args.scheduler == 'cosinerestarts':
        main_scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=total_iter//args.restart_epoch, T_mult=1 ,eta_min=args.min_lr)
    elif args.scheduler == 'multistep': # per designated epoch or iter.
        main_scheduler = MultiStepLR(optimizer=optimizer, milestones=[epoch * args.iters_per_epoch for epoch in args.milestones], gamma=args.decay_rate)
    elif args.scheduler == 'step': # per step size of epoch or iter.
        main_scheduler = StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.decay_rate)
    elif args.scheduler == 'exp':
        main_scheduler = ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)
    elif args.scheduler == 'onecycle':
        main_scheduler = OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=total_iter, three_phase=args.three_phase)
    else:
        raise NotImplementedError(f"{args.scheduler} not implemented yet")

    # warmup lr scheduler (per iter.)
    if args.warmup_epochs and args.scheduler != 'onecycle':
        if args.warmup_scheduler == 'linear':
            lr_lambda = lambda e: (e * (args.lr - args.warmup_lr) / warmup_iter + args.warmup_lr) / args.lr
            warmup_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        else:
            raise NotImplementedError(f"{args.warmup_scheduler} not implemented yet")

        scheduler = SequentialLR(optimizer=optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iter])

    else:
        scheduler = main_scheduler

    return optimizer, scheduler


def get_criterion_and_scaler(args):
    """
    - w/ CutMix or MixUp - you don't need to add smoothing loss, because CutMix or MixUp will add smoothing loss.
    - w/o CutMix or MixUp - you should need to add smoothing loss
    # fixme: without CutMix or MixUp smoothing or not doesn't matter??????
    """

    # criterion for training
    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BinaryCrossEntropy(label_smoothing=args.smoothing, bce_target=args.bce_target)

    # criterion for testing (w/o label smoothing)
    valid_criterion = nn.CrossEntropyLoss()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, valid_criterion, scaler
