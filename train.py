import torch
import time
import datetime
from args import get_args_parser
from setup import setup
from timm import create_model
from log import Result
from metric import Metric, reduce_mean, accuracy
from dataset import get_dataset, get_dataloader
from optimizer import get_optimizer_and_scheduler, get_criterion_and_scaler
from utils import get_ema_ddp_model, print_metadata, save_checkpoint, get_learning_rate



def run(args):

    # init DDP & logger
    setup(args)

    # load dataset & dataloader
    train_dataset, val_dataset = get_dataset(args)
    train_dataloader, val_dataloader = get_dataloader(train_dataset, val_dataset, args)

    # create model # fixme: change custom-resnet into timm
    model = create_model(model_name=args.model_name, num_classes=args.num_classes)
    model, ema_model, ddp_model = get_ema_ddp_model(model, args)

    # load optimizer & scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # load criterion(loss func) & scaler
    criterion, valid_criterion, scaler = get_criterion_and_scaler(args)

    # print metadata b/f training
    print_metadata(model, train_dataset, val_dataset, args)

    # control logic
    if args.resume:
        pass
        # start_epoch = resume_from_checkpoint() # fixme: implement resmue_from_checkpoint func,
    else:
        start_epoch = 0

    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    end_epoch = args.end_epoch if args.end_epoch else args.epochs

    if scheduler is not None and start_epoch:
        # fixme: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    if args.validate_only:
        validation(model, val_dataloader, valid_criterion, args, 'org')
        if args.ema:
            validation(ema_model.module, val_dataloader, valid_criterion, args, 'ema')
        return

    # train
    best_epoch = 0
    best_acc = 0
    top1_list = []
    top5_list = []
    start_time = time.time()

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(ddp_model if args.distributed else model, train_dataloader, optimizer, criterion, args, ema_model, scheduler, scaler, epoch)
        val_loss, top1, top5 = validation(ddp_model if args.distributed else model, val_dataloader, valid_criterion, args, 'org')

        if args.ema:
            eval_ema_metric = validation(ema_model.module, val_dataloader, valid_criterion, args, 'ema')

        if args.use_wandb:
            args.log({'train_loss': train_loss, 'val_loss': val_loss, 'top1': top1, 'top5': top5}, metric=True)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        top1_list.append(top1)
        top5_list.append(top5)

        if args.save_checkpoint and args.is_rank_zero:
            save_checkpoint(args.log_dir, model, ema_model, optimizer, scaler, scheduler, epoch, is_best=(best_epoch==epoch))

    # final summary
    if args.is_rank_zero:
        best_acc = round(float(best_acc), 4)
        top1 = round(float(sum(top1_list[-3:]) / 3), 4)
        top5 = round(float(sum(top5_list[-3:]) / 3), 4)
        duration = str(datetime.timedelta(seconds=int(time.time() - start_time))).split('.')[0]
        Result(args.output_dir).save_result(args, top1_list, top5_list,\
                                            dict(duration=duration, best_acc=best_acc, avg_top1_acc=top1, avg_top5_acc=top5))



@torch.inference_mode()
def validation(
    model,
    valid_dataloader,
    criterion,
    args,
    mode='org' # original or ema
):
    # create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, reduce_on_compute=True, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, reduce_on_compute=True, header='Top-5:')
    loss_m = Metric(reduce_every_n_step=args.print_freq, reduce_on_compute=True, header='Loss:')

    # model to validation mode
    model.eval()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=args.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            output = model(x)
            loss = criterion(output, y)

        top1, top5 = accuracy(output, y, topk=(1,5,))

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(args.iter_per_epoch))
            args.log(f"VALID({mode}): [{batch_idx:>{num_digits}}/{args.iter_per_epoch}] {batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]

    # average out acc. & loss in all GPUs for validation
    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()

    # print metric
    space = 16
    num_metric = 6
    args.log('-' * space * num_metric)
    args.log(("{:>16}" * num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-' * space * num_metric)
    args.log(f"{'VALID(' + mode + ')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-' * space * num_metric)

    return loss, top1, top5


def train_one_epoch(
        model,
        train_dataloader,
        optimizer,
        criterion,
        args,
        ema_model=None,
        scheduler=None,
        scaler=None,
        epoch=None
):
    # create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:') # loss of only local rank 0 GPU

    # model to train mode
    model.train()
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):

        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=args.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            output = model(x)
            loss = criterion(output, y)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss = loss / args.grad_accum # normalize loss to account for batch accumulation
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(args.iter_per_epoch))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{args.iter_per_epoch}] {batch_m} {data_m} {loss_m}") # TODO: add learning rate

        if batch_idx and ema_model and batch_idx % args.ema_update_step == 0:
            ema_model.update(model)

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # only seconds, not milli-sec or so
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=(batch_m.sum-data_m.sum))).split('.')[0]
    loss = loss_m.compute() # average loss of local rank 0 GPU

    # print metric
    space = 16
    num_metric = 5
    args.log('-' * space * num_metric)
    args.log(("{:>16}" * num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-' * space * num_metric)
    args.log(f"{'TRAIN(' + str(epoch) + ')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    args.log('-' * space * num_metric)

    return loss



if __name__ == '__main__':
    arg_parser = get_args_parser()
    args = arg_parser.parse_args()
    run(args)




