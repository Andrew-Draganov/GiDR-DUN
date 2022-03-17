import faiss
import torch
import torchvision.datasets as datasets
import time

mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
print(mnist_trainset.train_data.shape)
D = mnist_trainset.train_data.view(mnist_trainset.train_data.shape[0], -1).to(torch.float32)
print(D.shape)
D = D.numpy()
q = D

k = 20

res = faiss.StandardGpuResources()

t0 = time.time()
dist, N = faiss.knn_gpu(res, q, D, k)
t1 = time.time()

print("time", t1 - t0)
print(dist, N)
