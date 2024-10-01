import os
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel




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



class ModelEmaV2(nn.Module):
    """
    Model Exponential Moving Average V2
    copied from timm: https://timm.fast.ai/training_modelEMA

    This is applicable only to timm model
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def get_ema_ddp_model(model, args):
    model = model.to(args.device)

    # channels_last: (N, C, H, W) -> (N, H, W, C)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # compute mean & var across all GPUs (1. convert_sync_batchnorm(model) -> 2. DDP(model))
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # EMA model
    # evaluations that use averaged parameters sometimes produce significantly better results than the final trained values
    if args.ema:
        ema_model = ModelEmaV2(model, args.ema_decay)
    else:
        ema_model = None

    # DDP
    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ema_model, ddp_model


def print_metadata(model, train_dataset, test_dataset, args):
    title = 'INFORMATION'
    table = [
        ('Project Name', args.project_name),
        ('Project Administrator', args.who),
        ('Experiment Name', args.exp_name),
        ('Experiment Start Time', args.start_time),
        ('Experiment Model Name', args.model_name),
        ('Experiment Log Directory', args.log_dir)
    ]
    print_tabular(title, table, args)

    title = 'EXPERIMENT TARGET'
    table = [(target, str(getattr(args, target))) for target in args.exp_target]
    print_tabular(title, table, args)

    title = 'EXPERIMENT SETUP'
    table = [(target, str(getattr(args, target))) for target in [
        'train_size', 'train_resize_mode', 'random_crop_pad', 'random_crop_scale', 'random_crop_ratio',
        'test_size', 'test_resize_mode', 'center_crop_ptr',
        'interpolation', 'mean', 'std', 'hflip', 'auto_aug', 'cutmix', 'mixup', 'remode', 'aug_repeat',
        'model_name', 'ema', 'ema_decay', 'criterion', 'smoothing',
        'lr', 'epochs', 'optimizer', 'momentum', 'weight_decay', 'scheduler', 'warmup_epochs', 'batch_size',
        'grad_accum_step', 'total_batch_size'
    ]]
    print_tabular(title, table, args)

    # fixme: add flops

    title = 'DATA & MODEL'
    table = [
        ('Model Parameters(M)', count_parameters(model)),
        # FLOPs
        ('Number of Train Examples', len(train_dataset)),
        ('Number of Test Examples', len(test_dataset)),
        ('Number of Class', args.num_classes)
    ]
    print_tabular(title, table, args)

    title = 'TERMINOLOGY'
    table = [
        ('Batch', 'Time for 1 epoch in seconds'),
        ('Data', 'Time for loading data in seconds'),
        ('F+B+O', 'Time for Forward-Backward-Optimizer in seconds'),
        ('Top-1', 'Top-1 Accuracy'),
        ('Top-5', 'Top-5 Accuracy')
    ]
    print_tabular(title, table, args)

    args.log("-" * 81)


def count_parameters(model): # fixme: some parameters are missed...
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def print_tabular(title, table, args):
    title_space = int((81 - len(title)) / 2)
    args.log("-" * 81)
    args.log(" " * title_space + title)
    args.log("_" * 81)
    for (key, value) in table:
        args.log(f"{key:<25} | {value}")


def load_state_dict_from_checkpoint(checkpoint, key_list):
    for key in key_list:
        if checkpoint.get(key, None) is not None:
            return checkpoint.get(key)
    return None

def resume_from_checkpoint(checkpoint_path, model=None, ema_model=None, optimizer=None, scaler=None, scheduler=None):

    obj_key_list = [
        (model, ['state_dict', 'model']),
        (ema_model, ['state_dict_ema', 'model_ema', 'state_dict', 'model']),
        (optimizer, ['optimizer']),
        (scaler, ['scaler']),
        (scheduler, ['scheduler'])
    ]

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        for obj, key_list in filter(lambda x: x[0] is not None, obj_key_list):
            state_dict = load_state_dict_from_checkpoint(checkpoint, key_list)

            if state_dict:
                obj.load_state_dict(state_dict)
            elif 'state_dict' in key_list:
                obj.load_state_dict(checkpoint)
            else:
                raise ValueError(f'we can not find {key_list} in given checkpoint(dir={checkpoint_path})')

        return checkpoint.get('epochs', None) + 1

    else:
        raise ValueError(f'no file exist in given checkpoint_path argument(dir={checkpoint_path}')


def save_checkpoint(save_dir, model, ema_model, optimizer, scaler, scheduler, epoch, is_best=False):
    pairs = [('state_dict', model), ('state_dict_ema', ema_model),
             ('optimizer', optimizer), ('scaler', scaler), ('scheduler', scheduler)]
    checkpoint_dict = {k: v.state_dict() for k, v in pairs if v}
    checkpoint_dict['epochs'] = epoch

    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_{epoch}.pth'))
    torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_last.pth'))

    if is_best:
        torch.save(checkpoint_dict, os.path.join(save_dir, f'checkpoint_best.pth'))

    # remove unnecessary checkpoint file
    if os.path.exists(os.path.join(save_dir, f'checkpoint_{epoch - 10}.pth')):
        os.remove(os.path.join(save_dir, f'checkpoint_{epoch - 10}.pth'))


def get_learning_rate(optimizer):
    """
    Return learning rate history
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr




