# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils

from pathlib import Path

import time
import calendar
from codecarbon import OfflineEmissionsTracker

# def parse_args():

#     parser = argparse.ArgumentParser(description='Train Face Alignment')

#     parser.add_argument('--cfg', help='experiment configuration filename',
#                         required=True, type=str)

#     args = parser.parse_args()
#     update_config(config, args)
#     return args

def train(train_loader, val_loader):

    # args = parse_args()
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    model = 'CV-DeepLearning/hrnetv2_w18_imagenet_pretrained.pth'
    parser.add_argument('--cfg', default=model, help='experiment configuration filename')#,
                        # required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    print(args.cfg)
          
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    #CHRIS
    final_output_dir = final_output_dir + "/CodeCarbon_FINAL_3"
    final_output_dir = Path(final_output_dir)
    if not final_output_dir.exists():
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir()
    
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda() #MAX comment .cuda() if cpu

    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda() #MAX comment .cuda() if cpu

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file) #MAX , map_location=torch.device('cpu')) uncomment if cpu
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'], )
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )

    # dataset_type = get_dataset(config)
    # trainset = dataset_type(config, is_train=0),

    # train_loader = DataLoader(
    #     dataset=trainset,#dataset_type(config, is_train=0),
    #     batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
    #     shuffle=config.TRAIN.SHUFFLE,
    #     num_workers=config.WORKERS,
    #     pin_memory=config.PIN_MEMORY)

    # val_loader = DataLoader(
    #     dataset=dataset_type(config, is_train=1),
    #     batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=config.PIN_MEMORY
    # )

    #Startare il carbon footprint tracker
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme[5] < best_nme
        best_nme = min(nme[5], best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    #Stoppare il carbon footprint tracker
    emissions: float = tracker.stop()

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    total_emissions_file = os.path.join(final_output_dir,
                                          'emissions' + ts.__str__() + ".txt")
    
    f = open(total_emissions_file, "w")
    f.write(f"Emissions: {emissions} kg")
    f.close()

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state_FINAL_3.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


# if __name__ == '__main__':
#     main()










