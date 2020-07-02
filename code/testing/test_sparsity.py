from scipy.sparse import lil_matrix
import numpy as np

# m = 264 * 16
# n = (53-1) * 6 + 264 * 6
m = 5
n = 5
A = lil_matrix((m, n), dtype=int)

# i = np.arange(264 * 8)
i = np.arange(2)
j = np.arange(2)

for s in range(3):
    A[2 * i, 1] = 1
    # A[2 * i + 1, camera_indices * 6 + s] = 1
#
# for s in range(3):
#     A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
#     A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

print(A)
print(A.toarray())
