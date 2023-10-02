# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

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
    model = nn.DataParallel(model, device_ids=gpus)#.cuda() #MAX comment .cuda() if cpu

    #Una volta eseguire commentando load model e un'altra volta no (fino a riga 72)
    # load model

    state_dict = torch.load(args.model_file, map_location=torch.device('cpu')) #MAX, map_location=torch.device('cpu')) uncomment if cpu


    #print(type(state_dict))
    #print(state_dict)

    #if 'state_dict' not in state_dict.keys():
        # state_dict = state_dict['state_dict']
        #model.load_state_dict(state_dict)
    #else:
    model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config, is_train=3),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    nme, predictions = function.inference(config, test_loader, model)
    torch.save(predictions, os.path.join(final_output_dir, 'predictions_mc.pth'))

    csv_file = config.DATASET.DEMOSET
    images = pd.read_csv(csv_file)
    for idx in range(len(images)):
        image_path = os.path.join(config.DATASET.ROOT, images.iloc[idx, 0])
        plot_style = dict(marker='o',
                            markersize=2,
                            linestyle='-',
                            lw=2)

        input_img = io.imread(image_path)


        fig = plt.figure(figsize=plt.figaspect(0.3))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input_img)

        #In rosso stampiamo i punti predetti
        for i in range(68):
            ax.plot(predictions[idx][i][0],
                    predictions[idx][i][1],
                    color="red", **plot_style)
            
        plt.show()


if __name__ == '__main__':
    main()

