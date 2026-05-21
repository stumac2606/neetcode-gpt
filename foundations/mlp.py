import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        w = weights
        b = biases 
        h = x

        # Foward pass through each layer i in the network i.e length of the weights
        for i in range(len(w)):
            h = h @ w[i] + b[i]
            print(f"i: {i}")

            if i < len(w) - 1: # if the length of the layers in the network is less than the total length of the weights, calculate the activation. if it is not less than the total length, don't calculate the activation as we are at the final layer. it is len(weight) - 1 because the first value in the list is [0], and we dont want to calculate the activation for the output layer
                h = np.maximum(0, h)

        return np.round(h, 5)
        
