import os
import torch
from torch import distributed as dist

class Metric(object):
    def __init__(self, reduce_every_n_step=50, reduce_on_compute=False, header='', fmt='{value:.4f} ({avg:.4f})'):
        """
        Base Metric class supporting DDP setup

        :args
            reduce_every_n_step(int): call all_reduce every n step in DDP mode
            reduce_on_compute(bool): call all_reduce in compute() method
            fmt(str): format representing metric in string
        """
        self.distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.distributed:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.reduce_every_n_step = reduce_every_n_step
            self.reduce_on_compute = reduce_on_compute
        else:
            self.world_size = None
            self.reduce_every_n_step = self.reduce_on_compute = None

        self._reset(header, fmt)

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)

    def _reset(self, header, fmt):
        self.value = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update tensor data
        If not a tensor, then change it into a tensor
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()
        elif self.reduce_every_n_step and not isinstance(value, torch.Tensor):
            raise ValueError('reduce operation is allowed for tensor only')

        self.value = value
        self.sum += value * n
        self.n += n
        self.avg = self.sum / self.n

        if self.reduce_every_n_step and self.n % self.reduce_every_n_step == 0:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

    def compute(self):
        if self.reduce_on_compute:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

        return self.avg


def accuracy(output, target, topk=(1,)):
    """
    Computes the top-k for the specified values of k
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


def all_reduce_mean(val, world_size):
    """
    Collect value to each GPU
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM) # fixme: What's the difference b/t dist.all_reduce vs. dist.reduce()
    val = val / world_size
    return val


def reduce_mean(val, world_size):
    """
    Collect value to LOCAL 0 GPU
    """
    val = val.clone()
    dist.reduce(val, 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val