import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))
        print(f"std: {std}")

        w = torch.randn(fan_out, fan_in) * std
        print(f"w: {w}")

        # Round to 4 decimal places and return as nested list
        return torch.round(w, decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        torch.manual_seed(0)

        std = math.sqrt(2.0 / fan_in)

        w = torch.randn(fan_out, fan_in) * std 
        # Round to 4 decimal places and return as nested list
        return torch.round(w, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        torch.manual_seed(0)

        stds = []
        weights = []
        for i in range(num_layers):
            if i == 0:
                fan_in = input_dim
            else:
                fan_in = hidden_dim
            fan_out = hidden_dim
        
            if init_type == 'xavier':
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / fan_in)
            else:
                std = 1.0

            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)

            
        x = torch.randn(1, input_dim)
        
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)

            stds.append(round(x.std().item(), 2))

        

        return stds
