import numpy as np
arr = np.array([[1, 1, 1], [0, 0, 0]])
arr[np.arange(2), [0, 2]] = [4, 5]
print(arr)
