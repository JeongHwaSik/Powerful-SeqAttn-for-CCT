import os
import torch
import wandb

def init_wandb(args):

    # 🐝 setup wandb.ai
    if args.wandb and args.is_rank_zero:
        wandb.init(
            project=args.proj,
            name=args.expname,
            config=args,
            reinit=True
        )

def init_distributed_mode(args):
    os.environ["INITIALIZED"] = 'TRUE'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # fixme : GPU device 설정???
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        torch.distributed.init_process_group(
            backend = args.dist_backend,
            init_method = args.dist_url,
            world_size = args.world_size,
            rank = args.rank
        )


    else:
        args.distributed = False
        args.world_size = 1
        args.gpu = 0

    args.is_rank_zero = (args.gpu == 0)
    allow_print_to_master(args.is_rank_zero)

    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')

    if args.distributed:
        torch.distributed.barrier() # wait until all processes are here


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