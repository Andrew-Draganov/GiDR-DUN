import numpy as np
from umap_ import UMAP

np.random.seed(12345)
a = np.random.random([5000, 200])

dr = UMAP(random_state=12345)
print('fitting...')
dr.fit(a)
print('transforming...')
b = dr.transform(a)

correct_outputs = np.loadtxt('nndescent_outputs.txt')

try:
    np.testing.assert_allclose(b, correct_outputs)
    print("Equivalent to original UMAP implementation")
except AssertionError:
    raise AssertionError("Not equivalent to the original UMAP implementation!")
