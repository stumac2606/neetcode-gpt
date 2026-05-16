import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        print(f"N: {N}")
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N
        #gradient = self.get_derivative(predictions, Y, len(X), X, weights)

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        weights = initial_weights
        print(f"weights: {weights}")
        for i in range(num_iterations): # needed to find renge() function
        
        #   1. Compute predictions with get_model_prediction(X, weights)
            predictions = self.get_model_prediction(X, weights)
            print(f"predictions: {predictions}")
        #   2. For each weight index j, compute gradient with get_derivative()
            for j in range(len(weights)):
                gradient = self.get_derivative(predictions, Y, len(X), X, j)

        #   3. Update: weights[j] -= learning_rate * gradient
                weights[j] -= self.learning_rate * gradient
            print(f"i: {i}")
        final_weights = weights
        # Return np.round(final_weights, 5)
        return np.round(final_weights, 5)
