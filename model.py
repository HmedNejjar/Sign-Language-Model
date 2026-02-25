import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import Tensor

class SLModel(nn.Module):
    def __init__(self, num_classes: int, input_size:int = 225, hidden_size: int = 128, num_layers:int = 2,num_neurons:int = 64, dropout: float = 0.3) -> None:
        """
        Args:
            num_classes: number of classes the model needs to predict from
            input_size: number of features per neuron
            hidden_size: number of predictions per neuron
            num_layers: number of layers in the model's architecture
            num_neurons: number of neurons per cortical columnm network (CCN)
            dropout: % of neurons to turn off during training
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        
        # Defining the model's parameters
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias= True, batch_first= True, dropout= dropout if num_layers > 1 else 0)
        
        # Classifier head
        """How it works:
        nn.Sequential(): Allows data processing following order in the method
        nn.Linear(hidden_size, num_neurons): Linear operation on the input, output according to the number of neurons
        nn.ReLU(): Immediately processes the output of the first linear layer. If a neuron's value is negative, ReLU turns it into zero.
        nn.Dropout(dropout): Forcing the learning before final step by turning off 'dropout'% of neurons
        nn.Linear(num_neurons, num_classes): Linear operation on input from neurons, output acording to number of classes we have
        """
        self.fc = nn.Sequential(nn.Linear(hidden_size, num_neurons), nn.ReLU(), nn.Dropout(dropout), nn.Linear(num_neurons, num_classes))
        
    def forward(self, X:Tensor) -> Tensor:
        # Initialize hidden and cell states
        h0 = torch.zeros((self.num_layers, X.size(0), self.hidden_size)).to(X.device)
        c0 = torch.zeros((self.num_layers, X.size(0), self.hidden_size)).to(X.device)
        
        # Forward Pass through the model
        out, _ = self.lstm(X, (h0, c0))     # out.shape = (batch_size, sequence_len, hidden_size)
        out = self.fc(out[:, -1, :])    # passing only the last frame out[:, -1, :], by that time h_60 would have summarized the video data
        
        return out
    
if __name__ == "__main__":
    # Quick test to check if shapes are correct
    model = SLModel(num_classes=2000) # example with 2000 words
    dummy_input = torch.randn(32, 60, 225)      # (Batch, Frames, Features)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")      # Should be [32, 2000]