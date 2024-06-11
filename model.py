import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List

class Model(nn.Module):
    def __init__(self, input_shape, choose: List[str]):
        super(Model, self).__init__()
        self.height, self.width = input_shape
        
        # Convolutional layers
        self.ConvLayer = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)),
                ('batchnorm1', nn.BatchNorm2d(32)),
                ('relu1', nn.LeakyReLU()),
                ('dropout1', nn.Dropout(0.25)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                
                ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                ('batchnorm2', nn.BatchNorm2d(64)),
                ('relu2', nn.LeakyReLU()),
                ('dropout2', nn.Dropout(0.25)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                
                ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
                ('batchnorm3', nn.BatchNorm2d(128)),
                ('relu3', nn.LeakyReLU()),
                ('dropout3', nn.Dropout(0.25)),
                ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ])
        )
        
        # Calculate the size after convolutions to set the size of the fully connected layer
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        conv_output = self.ConvLayer(dummy_input)
        conv_output_size = conv_output.numel()
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(conv_output_size, 1024), 
            nn.Dropout(0.4),
        )
        
        # Output layers
        self.output_layers = nn.ModuleDict()
        if "ID" in choose:
            self.output_layers["ID"] = nn.Sequential(
                nn.Linear(1024, 600),
                nn.Dropout(0.4),
            )
        if "GENDER" in choose:
            self.output_layers["GENDER"] = nn.Sequential(
                nn.Linear(1024, 1),
                nn.Dropout(0.5),
            )
        if "HAND" in choose:
            self.output_layers["HAND"] = nn.Sequential(
                nn.Linear(1024, 1),
                nn.Dropout(0.5),
            )
        if "FINGER" in choose:
            self.output_layers["FINGER"] = nn.Sequential(
                nn.Linear(1024, 5),
                nn.Dropout(0.5),
            )
        
    def forward(self, x): 
        x = self.ConvLayer(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        
        outputs = {}
        for name, layer in self.output_layers.items():
            outputs[name] = layer(x)
        
        return outputs

