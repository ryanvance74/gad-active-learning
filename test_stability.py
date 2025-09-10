import numpy as np
from optimal_design_sub_mod import sherman_update, get_next_point

dataset = np.array([[5,1,2,3,4],
                    [1.5,-2,7.9,4,5],
                    [-0.2,-0.9,0.8,7,-5],
                    [8,4,2,3,-0.1],
                    [2,2,3,-1,4]])
eps = 0.001
Vinv = eps * np.eye(dataset.shape[1])
V = (1./eps) * np.eye(dataset.shape[1])    
# V = 1 / 0.001 * np.eye(dataset.shape[1])
next_point = get_next_point(V, dataset)

from_inv = np.linalg.inv(Vinv + np.outer(next_point, next_point))

from_sherman = sherman_update(V, next_point)

print("DATASET\n", dataset)
print("POINT CHOSEN\n", next_point.T)
print("WITH LINALG.INV\n", from_inv, "\nWITH SHERMAN UPDATE\n", from_sherman, "\nDIFFERENCE MATRIX\n", from_inv - from_sherman)

print("NORM OF DATASET: ", np.linalg.norm(dataset, 'fro'))
print("NORM OF DIFFERENCE MATRIX: ", np.linalg.norm(from_inv - from_sherman, 'fro'))
