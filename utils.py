import os
import torch
import torch.nn as nn
from torch import distributed as dist


class Metric(object):
    def __init__(self, reduce=False, header=''):
        self._reset()
        self.distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        # Need reduce computation for DDP or not
        self.reduce = reduce

        if self.distributed and reduce:
            self.world_size = int(os.environ['WORLD_SIZE'])
        else:
            self.world_size = None


    def _reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        """
        Update tensor data
        If not a tensor, then change it into a tensor
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()

        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

        if self.distributed and self.reduce:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.count

def all_reduce_mean(val, world_size):
    """
    Collect value to each GPU
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM) # fixme: What's the difference b/t dist.all_reduce vs. dist.reduce()
    val = val / world_size
    return val


def count_params(model: nn.Module):
    """
    Count the number of parameters in a model
    """
    return sum([m.numel() for m in model.parameters()])


# fixme: doesn't work? Found at least two devices ERROR
def deepspeed_profile(model, input_size=(3, 32, 32), batch_size=1, detailed=False):
    from deepspeed.profiling.flops_profiler import get_model_profile
    """
    Measure FLOPs(MACs) using deepspeed profile
    """
    _, macs, _ = get_model_profile(
        model = model,
        input_shape = (batch_size, *input_size),
        print_profile = detailed, # prints the model graph with the measured profile attached to each module
        detailed = detailed, # print the detailed profile
        warm_up = 0, # the number of warm-ups before measuring the time of each module
        as_string = True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file = None, # path to the output file. If None, the profiler prints to stdout
        ignore_modules = None # the list of modules to ignore in the profiling
    )
    return macs
