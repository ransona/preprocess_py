import numpy as np
from scipy.sparse import csr_matrix
import sparse

# Create a 3D numpy array
arr = np.zeros((3, 4, 5))

# Set some values to non-zero
arr[0, 1, 2] = 3
arr[2, 1, 4] = 5

# Convert the 3D numpy array to a sparse matrix in COO format
sparse_matrix = sparse.COO(arr)

# Print the sparse matrix
print(sparse_matrix)

