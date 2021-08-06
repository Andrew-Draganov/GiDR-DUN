import numpy as np
from umap_ import UMAP

a = np.ones([200, 200])
dr = UMAP(random_state=12345)

b = dr.fit_transform(a)

correct_outputs = np.loadtxt('test_outputs.txt')

try:
    np.testing.assert_allclose(b, correct_outputs)
    print("Equivalent to original UMAP implementation")
except AssertionError:
    raise AssertionError("Not equivalent to the original UMAP implementation!")
