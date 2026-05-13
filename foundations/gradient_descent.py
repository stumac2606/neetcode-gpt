class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        x = init
        # Derivative:         f'(x) = 2x
        for i in range(iterations):
            grad = 2 * x
            print(f"Gradient = {grad}")
        # Update rule:        x = x - learning_rate * f'(x)
            x = x - learning_rate * grad
            print(f"Value: {x}")
        # Round final answer to 5 decimal places
        return round(x, 5)
