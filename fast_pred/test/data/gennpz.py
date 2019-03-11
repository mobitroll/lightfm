import numpy as np
from scipy import sparse as sp

a = np.array([[1,0,0], [0,1,0], [0,0,1]])

sp.save_npz("sptest.npz", sp.csr_matrix(a))

np.savez_compressed("nptest.npz", a=a)