import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
from collections import OrderedDict
# from train import train
from test import test
import torch
from torch.utils.data import DataLoader, random_split
from lib.datasets import get_dataset
from lib.config import config, update_config
import numpy as np
import lib.models as models
''''''
import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.backends.cudnn as cudnn
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

''''''
NUM_CLIENTS = 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def test():
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    model = 'D:/Tesi master/Federated-learning/CV-DeepLearning/hrnetv2_w18_imagenet_pretrained.pth'
    parser.add_argument('--cfg', default=model,help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', default='D:/Tesi master/Federated-learning/CV-DeepLearning/model_best_FINAL_3.pth', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)

    print(args.cfg)
    print(args.model_file)

    # cfg = "experiments/300w/face_alignment_300w_hrnet_w18.yaml"
    # model_file = "HR18-300W.pth"

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

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


def train(train_loader, val_loader):

    # args = parse_args()
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    model = 'D:/Tesi master/Federated-learning/CV-DeepLearning/hrnetv2_w18_imagenet_pretrained.pth'
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

    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.determinstic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED

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

    dataset_type = get_dataset(config)
    trainset = dataset_type(config, is_train=0),

    train_loader = DataLoader(
        dataset=trainset,#dataset_type(config, is_train=0),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config, is_train=1),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

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

def get_parameters(net) -> List[np.ndarray]:
    """get the updated model parameters from the local model"""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """Update the local model with parameters received from the server"""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    """Client subclass"""
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
# def client_fn(train_loaders, val_loaders, cid: str) -> FlowerClient: # cid: Client ID
def client_fn(cid: str) -> FlowerClient: # cid: Client ID
    """Create a Flower client representing a single organization."""


    # Load model
    net = models.get_face_alignment_net(config)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = train_loaders[int(cid)]
    valloader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

def load_Dataset(config):
    """Loads the and prepare the dataset for federated learning simulation, partitioning it among clients"""
    gpus = list(config.GPUS)
    dataset_type = get_dataset(config)
    train_set = dataset_type(config, is_train=0)
    val_set = dataset_type(config, is_train=1)
    
    # Split training and validation set into NUM_CLIENTS partitions to simulate the individual dataset
    train_set = ds_partition(train_set, NUM_CLIENTS)
    val_set   = ds_partition(val_set, NUM_CLIENTS)

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for ds_train, ds_val in zip(train_set,val_set):
        train_loaders.append(DataLoader(
            ds_train, 
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus), 
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
            ))
        val_loaders.append(DataLoader(
            ds_val, 
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
            ))
    return train_loaders, val_loaders

def ds_partition(set, num_clients):
    """Split the dataset into num_clients partitions"""
    partition_size = len(set) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] = lengths[-1]+len(set)-partition_size*num_clients
    set = random_split(set, lengths, torch.Generator().manual_seed(42))
    return set

def main():

    train_loaders, val_loaders = load_Dataset(config)

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit            =1.0,           # 1.0 : Sample 100% of available clients for training
        fraction_evaluate       =0.5,           # 0.5 : Sample 50% of available clients for evaluation
        min_fit_clients         =NUM_CLIENTS,   # 2   : Never sample less than NUM_CLIENTS clients for training
        min_evaluate_clients    =1,             # 1   : Never sample less than 1 client for evaluation
        min_available_clients   =NUM_CLIENTS,   # Wait until all clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,#(train_loaders, val_loaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == '__main__':
    main()