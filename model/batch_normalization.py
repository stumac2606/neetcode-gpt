import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # # convert the lists from input into arrays so we can manipluate and perfrom calculations on them 
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)

        # # convert to arrays and make sure we have float64 to get 64 bit values so we dont lose accracy in decimal values and provides high precison 
        running_mean = np.array(running_mean, dtype = np.float64)
        running_var = np.array(running_var, dtype = np.float64)
        # During inference: normalize using running stats (no batch stats needed)
        if training:
            # # calculate batch mean during training using np.mean 
            # # we use axis = 0 because a 2d array is made up of (rows, columns), so this essentially says axis = rows so we are perfroming the mean and var calculations across the rows so we get the mean of the rows in a feature
            batch_mean = np.mean(x, axis = 0)
            batch_var = np.var(x, axis = 0)
            # Apply affine transform: y = gamma * x_hat + beta
            # x_hat will now have a mean of exactly 0 and a variance of exactly 1.
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
        
        else:
             x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        out = gamma * x_hat + beta

        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        return (np.round(out, 4).tolist(), np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist())