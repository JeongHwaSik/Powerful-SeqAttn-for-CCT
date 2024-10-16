import os
import gc
import wandb
import logging
import glob
import numpy
import random
import torch
from pathlib import Path
from functools import partial
from datetime import datetime


def make_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y/%m/%d %H:%M")

    # logging at console (StreamHandler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console) # attach to logger

    # logging at file (FileHandler)
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) # attach to logger

    return logger


def log(msg, metric=False, logger=None):
    """
    function that do write log info.
    """
    if logger:
        if metric:
            wandb.log(msg)
        else:
            logger.info(msg)


def init_logger(args):
    if args.resume:
        args.exp_name = Path(args.checkpoint_path).resolve().parent.name
        print(args.exp_name)
    else:
        if args.exp_name is None:
            args.exp_name = '_'.join(str(getattr(args, target)) for target in args.exp_target)
        args.version_id = len(list(glob.glob(os.path.join(args.output_dir, f'{args.exp_name}_v*'))))
        args.exp_name = f'{args.exp_name}_v{args.version_id}'

    args.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.log_dir = os.path.join(args.output_dir, args.exp_name)
    args.text_log_path = os.path.join(args.log_dir, 'log.txt')
    args.best_weight_path = os.path.join(args.log_dir, 'best_weight.pth')

    if args.is_rank_zero:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        args.logger = make_logger(args.text_log_path)
        # 🐝
        if args.use_wandb:
            wandb.init(project=args.project_name, name=args.exp_name, id=args.exp_name, config=args, reinit=True, resume="allow")
    else:
        args.logger = None

    args.log = partial(log, logger=args.logger)


def allow_print_to_master(is_rank_zero):
    """
    change default print() function into print only for rank 0 GPU
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force or is_rank_zero:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def check_need_init():
    """
    to prevent re-initialization for multi-train
    """
    if os.environ.get('INITIALIZED', None):
        return False
    else:
        return True


def init_distributed_mode(args):
    os.environ["INITIALIZED"] = 'TRUE'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f'{datetime.now().strftime("[%Y/%m/%d %H:%M]")} ', end='')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # singe node: len(global_rank) == world_size == num of GPUs
        # single node: global_rank == local_rank
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        torch.distributed.init_process_group(
            backend = args.dist_backend,
            init_method = args.dist_url,
            world_size = args.world_size,
            rank = args.rank
        )

    else:
        print(f'| Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.gpu = 0

    args.is_rank_zero = (args.gpu == 0)
    allow_print_to_master(args.is_rank_zero)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')

    if args.distributed:
        torch.distributed.barrier() # wait until all processes are here



def seed_worker(worker_id):
    """
    you can use DataLoader(..., worker_init_fn=seed_worker) for fixed seed in DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def clear(args):
    # clear up GPU memory
    torch.cuda.empty_cache()

    # clear up CPU memory
    gc.collect()

    # close logger & finish wandb 🐝
    if args.is_rank_zero:
        handlers = args.logger.handlers[:]
        for handler in handlers:
            args.logger.removeHandler(handler)
            handler.close()
        if args.use_wandb:
            wandb.finish(quiet=True)


def setup(args):
    # init DDP
    if check_need_init:
        init_distributed_mode(args)
    # init logger
    init_logger(args)

    if args.deterministic_seed:
        # cuDNN - fixed convolution benchmark
        torch.backends.cudnn.benchmark = False
        # other functions which acts as non-deterministic
        torch.use_deterministic_algorithms(True)
        # fixed seed
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        # dataloader - fixed seed
        args.seed_worker = seed_worker
    else:
        torch.backends.cudnn.benchmark = True
        args.seed_worker = None


if __name__ == '__main__':
    """
    e.g. CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 setup.py
    This will run the process on GPU 0 and GPU 1, respectively.
    
    output:
    0
    2
    0
    1
    2
    1
    """
    print(os.environ['RANK'])
    print(os.environ['WORLD_SIZE'])
    print(os.environ['LOCAL_RANK'])

        
