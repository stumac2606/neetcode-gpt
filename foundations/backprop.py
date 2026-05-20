import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        print(f"z: {z}")

        y_hat = 1/(1 + np.exp(-z))
        print(f"y_hat: {y_hat}")

        # Loss: L = 0.5 * (y_hat - y_true)^2
        L = 0.5 * ((y_hat - y_true)**2)
        print(f"loss: {L}")

        # Gradient 
        dL_dW = (y_hat - y_true) * (y_hat*(1 - y_hat)) * x
        print(f"dL_dW: {dL_dW}")

        dL_dB = np.dot((y_hat - y_true), (y_hat*(1 - y_hat)))
        print(f"dL_dB: {dL_dB}")
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return np.round(dL_dW, 5), round(dL_dB, 5)
