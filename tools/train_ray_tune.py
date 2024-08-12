import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

import ray
from ray.tune import CLIReporter
from ray import tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from easydict import EasyDict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/opv2v_models/collective_pv_rcnn_plusplus.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='HyperOpt', help='extra tag for this experiment')
    parser.add_argument('--device', type=int, default=0, help='device for non-distributed training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=80, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=80, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=True, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

from pcdet.datasets import build_dataloader as original_build_dataloader

#Global data directory

def load_data(cfg, args, dist_train, logger, training=True):
    cfg.DATA_CONFIG.DATA_PATH = '/local/teufelsv/collectivepcdet/data/OPV2V'
    train_set, train_loader, train_sampler = original_build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, 
        workers=args.workers,
        logger=logger,
        training=training,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    return train_set, train_loader, train_sampler

def training_function(config, data_dir=None):
    args, cfg = config['args'], config['cfg']
    cfg = EasyDict(cfg)

    # Regular training as in the case without Hyperparameter tweaking
    
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
        torch.cuda.set_device(args.device)
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = load_data(cfg, args, dist_train, logger, training=True)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True) # todo: fix unused params
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )


    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        args=args
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    

if __name__ == "__main__":
    ray.init(log_to_driver=False)

    args, cfg = parse_config()

    # specify search space
    cfg.OPTIMIZATION.LR = tune.loguniform(1e-5, 1e-1)

    # specify fixed params
    cfg.OPTIMIZATION.NUM_EPOCHS = 80
    cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = 6
    gpus_per_trial = 1
    cpus_per_trial = 42
    num_samples = 16
    args_cfg = {'args': args, 'cfg': cfg}
    
    scheduler = ASHAScheduler(metric="AP70", mode="max", max_t=cfg.OPTIMIZATION.NUM_EPOCHS, grace_period=20, reduction_factor=2)
    search_alg = OptunaSearch(metric="AP70", mode="max")
    reporter = CLIReporter(metric_columns=["AP70", "training_iteration"])

    
    
    result = tune.run(
        training_function,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        config=args_cfg,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        log_to_file=True
    )

    best_trial = result.get_best_trial("AP70", "max", "all")
    print(f"Best trial config: {best_trial.config['cfg']['OPTIMIZATION']}")
    print(f"Best trial final AP70: {best_trial.last_result['AP70']}")