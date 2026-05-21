import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        
        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        print(f"X shape: {x.shape}")
        print(f"w1 shape: {W1.shape}")
        print(f"w2 shape: {W2.shape}")
        print(f"b1 shape: {b1.shape}")
        print(f"b2 shape: {b2.shape}")
        print(f"y_true shape: {y_true.shape}")
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        
        #1 Foward to get predictions on first layer
        z1 = np.matmul(x, W1.T) + b1
        print(f"Prediction1: {z1}")
        #2 predictions of first neuron using ReLU
        a1 = np.maximum(0, z1)
        print(f"activation 1: {a1}")
        

        #2 Foward to get predictions on second layer 
        z2 = np.matmul(a1, W2.T) + b2 # what does .T do?

        #3 assign predictions from output 
        y_hat = z2

        print(f"Predictions 2: {y_hat}")

        # Loss: MSE = mean((predictions - y_true)^2), loss from the predictions caluclated
        L = np.mean((y_hat - y_true)**2)
        print(f"Loss: {L}")

        n = len(y_true) if y_true.ndim > 0 else 1
        # output layer
        dL_dz2 = 2*(y_hat - y_true) / n # its length of y_true because the MSE calculation divides by y_true 
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2

        # Back prop through w2 to get graident on a1
        dL_da1 = np.matmul(W2.T, dL_dz2)

        # Back prop through relu 
        dL_dz1 = dL_da1 * (z1 > 0)


        # Hidden layer 
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'loss' : round(float(L), 4),
            'dW1' : np.round(dL_dW1 + 0.0, 4).tolist(),
            'db1' : np.round(dL_db1 + 0.0, 4).tolist(),
            'dW2' : np.round(dL_dW2 + 0.0, 4).tolist(),
            'db2' : np.round(dL_db2 + 0.0, 4).tolist(),
        }







        

        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return y_hat2
