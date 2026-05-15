import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        x_pred = np.dot(X, weights)
        print(f"x_pred: {np.round(x_pred, 5)}")

        return np.round(x_pred, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        MSE = np.mean((model_prediction - ground_truth)**2)

        return round(MSE, 5)
