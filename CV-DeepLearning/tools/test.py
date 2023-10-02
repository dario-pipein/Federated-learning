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
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function

import time
import calendar
from codecarbon import OfflineEmissionsTracker


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    print(args.cfg)
    print(args.model_file)

    # cfg = "experiments/300w/face_alignment_300w_hrnet_w18.yaml"
    # model_file = "HR18-300W.pth"

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda() #MAX comment .cuda() if cpu

    #Una volta eseguire commentando load model e un'altra volta no (fino a riga 72)
    # load model
    state_dict = torch.load(args.model_file) #MAX, map_location=torch.device('cpu')) uncomment if cpu

    #print(type(state_dict))
    #print(state_dict)

    #if 'state_dict' not in state_dict.keys():
        # state_dict = state_dict['state_dict']
        #model.load_state_dict(state_dict)
    #else:
    model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config, is_train=2),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    #Startare il carbon footprint tracker
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    tracker.start()

    nme, predictions = function.inference(config, test_loader, model)

    #Stoppare il carbon footprint tracker
    emissions: float = tracker.stop()

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    total_emissions_file = os.path.join(final_output_dir,
                                          'emissions_test_FINAL_3B' + ts.__str__() + ".txt")
    
    f = open(total_emissions_file, "w")
    f.write(f"Emissions: {emissions} kg")
    f.close()

    torch.save(predictions, os.path.join(final_output_dir, 'predictions_FINAL_3B.pth'))


if __name__ == '__main__':
    main()

