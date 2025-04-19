import fastf1
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


"""
    Steps:
        1. Get the qualifying data of a given driver, example Leclerc
        2. Train a model based on all their laptimes and predict what their qualifying lap will be
            -> 
"""