import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple
from collections import OrderedDict
from tools.train import train
from tools.test import test
import torch
import numpy as np
import lib.models as models

NUM_CLIENTS = 2
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train.main(self.net, self.trainloader)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
def client_fn(cid: str) -> FlowerClient: # cid: Client ID
    """Create a Flower client representing a single organization."""

    # Load model
    net = models.get_face_alignment_net()

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = train_loaders[int(cid)]
    valloader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)

def main():
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit            =1.0,   # Sample 100% of available clients for training
        fraction_evaluate       =0.5,   # Sample 50% of available clients for evaluation
        min_fit_clients         =10,    # Never sample less than 10 clients for training
        min_evaluate_clients    =5,     # Never sample less than 5 clients for evaluation
        min_available_clients   =10,    # Wait until all 10 clients are available
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == '__main__':
    main()