import argparse
import os

import gym
import numpy as np
from torch import nn

def s2b(val):
    return val.lower() in ('yes', 'true', 't', '1')

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25_000, help="total timesteps of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x:  bool(s2b(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backend.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(s2b(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(s2b(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with MLFlow")
    parser.add_argument("--mlflow-server", type=str, default=None, help="the host for a mlflow server")
    parser.add_argument("--capture-video", type=lambda x: bool(s2b(x)), default=False, nargs="?", const=True,
        help="capture videos of the agent performances (stored in `videos`)")

def layer_init(layer, std=np.sqrt(2), bias_const=0.0) -> None:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer