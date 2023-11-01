import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
from collections import OrderedDict
from train import train
from test import test
import torch
from torch.utils.data import DataLoader, random_split
from lib.datasets import get_dataset
from lib.config import config, update_config
import numpy as np
import lib.models as models

NUM_CLIENTS = 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    
def client_fn(train_loaders, val_loaders, cid: str) -> FlowerClient: # cid: Client ID
    """Create a Flower client representing a single organization."""

    # Load model
    net = models.get_face_alignment_net()

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
    val_set = ds_partition(val_set, NUM_CLIENTS)

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
        min_fit_clients         =10,            # 10  : Never sample less than 10 clients for training
        min_evaluate_clients    =5,             # 5   : Never sample less than 5 clients for evaluation
        min_available_clients   =NUM_CLIENTS,   # Wait until all clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn(train_loaders, val_loaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == '__main__':
    main()