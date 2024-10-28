import torch
from logger import Logger
from torch import nn

logger: Logger = Logger().get_logger()

"""
We define our neural network by subclassing nn.Module
"""
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")
