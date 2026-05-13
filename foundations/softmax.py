import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp

        z_stable = np.exp(z - np.max(z))

        softmax = z_stable / np.sum(z_stable)

        return np.round(softmax, 4)
        
        
