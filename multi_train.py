import os
import sys
import argparse
from pathlib import Path
from args import get_args_parser
from traceback import print_exc
from setup import clear
from train import run

# CUDA_VISIBLE_DEVICES=0 python3 multi_train.py cifar100 -m resnet50
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 multi_train.py cifar100 -m resnet50

# TODO: add '--resume --checkpoint-path {args.output_dir}/{args.model_name}_{args.exp_name}_{version}/checkpoint_last.path' for resume from the checkpoint
setting_dict = dict(
    cifar100="data --dataset-name CIFAR100 --train-size 32 32 --train-resize-mode RandomResizedCrop --random-crop-scale 0.8 1.0 --test-size 32 32 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --auto-aug rand-m9-mstd0.5-inc1 --cutmix 1.0 --mixup 0.8 --remode 0.25 --drop-path-rate 0.0 --smoothing 0.1 --batch-size 256 --grad-accum-step 1 --epoch 600 --lr 5e-4 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epoch 10 --optimizer adamw --weight-decay 6e-2 --fused --scheduler cosine --num-workers 4 --pin-memory --amp --channels-last --save-checkpoint",
    # imagenet="data/imageNet --dataset_type ImageFolder --train-size 224 224 --train-resize-mode RandomResizedCrop --random-crop-scale 0.8 1.0 --test-size 224 224 --center-crop-ptr 1.0 --interpolation bicubic --mean 0.5071 0.4867 0.4408 --std 0.2675 0.2565 0.2761 --auto-aug rand-m9-mstd0.5-inc1 --cutmix 1.0 --mixup 0.8 --remode 0.25 --drop-path-rate 0.0 --smoothing 0.1 -b 256 --epoch 300 --lr 5e-4 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epoch 10 --optimizer adamw --weight-decay 6e-2 --scheduler cosine -j 4 --pin-memory --amp --channels-last --save-checkpoint",
)

def get_multi_args_parser():
    """
    efficiently run code

    for every setup:
        for every models:
            ~
    """
    parser = argparse.ArgumentParser(description='multi-args for DL classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['resnet50'], help='list of model names to train')
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda device(s)')
    parser.add_argument('-o', '--output-dir', type=str, default='log', help='log dir')
    parser.add_argument('-p', '--project-name', type=str, default='DL classification', help='project name used for wandb project name')
    parser.add_argument('-w', '--who', type=str, default='RyanJ', help='project owner')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb')

    # use only one model for resuming checkpoint
    # TODO: change to automatically configure checkpoint_path by model_name
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint path (need to specify checkpoint_path)')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='checkpoint path for resuming')
    return parser


def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def save_arguments(multi_args, is_master):
    if is_master:
        print("Multiple Train Setting")
        print(f" - model (num={len(multi_args.model_name)}): {', '.join(multi_args.model_name)}")
        print(f" - setting (num={len(multi_args.setup)}): {', '.join(multi_args.setup)}")
        print(f" - cuda: {multi_args.cuda}")
        print(f" - output dir: {multi_args.output_dir}")

        # TODO: add argument.txt file
        Path(multi_args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(multi_args.output_dir, 'last_multi_args.txt'), 'wt') as f:
            f.write(" ".join(sys.argv))



if __name__ == '__main__':
    is_master = os.environ.get('LOCAL_RANK', None) is None or int(os.environ['LOCAL_RANK']) == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    # save_arguments(multi_args, is_master)
    prev_args = None

    for setup in multi_args.setup:
        args_parser = get_args_parser()
        args = args_parser.parse_args(setting_dict[setup].split(' '))
        pass_required_variable_from_previous_args(args, prev_args)
        for model_name in multi_args.model_name:
            args.setup = setup
            args.exp_name = f"{model_name}_{setup}"
            args.model_name = model_name
            for option_name in ['cuda', 'output_dir', 'project_name', 'who', 'use_wandb', 'resume', 'checkpoint_path']:
                exec(f"args.{option_name} = multi_args.{option_name}")
            try:
                run(args)
            except:
                print(print_exc())
            clear(args)
        prev_args = args


