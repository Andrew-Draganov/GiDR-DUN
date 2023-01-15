import torch
import numpy as np
from GDR.nndescent.py_files.pynndescent_ import NNDescent

n_vertices = 100
X = torch.normal(0, 1, [n_vertices, 50])
Y = torch.normal(0, 1, [n_vertices, 2], requires_grad=True)

x_sq_dists = torch.sum(torch.square(torch.unsqueeze(X, 0) - torch.unsqueeze(X, 1)), axis=-1)
y_vectors = torch.unsqueeze(Y, 0) - torch.unsqueeze(Y, 1)
y_sq_dists = torch.sum(torch.square(y_vectors), axis=-1)

p_ij = torch.exp(-1 * x_sq_dists)
P = p_ij / torch.sum(p_ij)

inverse_dist = 1 / (1 + y_sq_dists)
Z = torch.sum(inverse_dist)
Q = inverse_dist / Z

### AUTOMATIC GRAD
L_mat = (P - Q) ** 2
L = torch.sum(L_mat)
L.backward()
torch_grad = Y.grad.detach().numpy()

### GRAD BY HAND
# Old gradient calculation
# q2_q3 = -Q ** 2 + 2 * Q ** 3
# attr = -P * Z * q2_q3
# rep = Q * Z * q2_q3

# Correct gradient calculation
k_neq_l = torch.sum(Q * Q - P * Q)
left_sum = k_neq_l * Q * Q
right_sum = Q * Q * (P - Q)
hand_grad = 8 * Z * (right_sum + left_sum)
hand_grad = torch.stack([hand_grad, hand_grad], axis=-1) # Set up shape to broadcast with y_vectors on next line
hand_grad *= y_vectors
hand_grad = torch.sum(hand_grad, axis=0)
hand_grad = hand_grad.detach().numpy()

# Print cosine similarity
torch_norm = np.sqrt(np.sum(np.square(torch_grad), axis=-1))
hand_norm = np.sqrt(np.sum(np.square(hand_grad), axis=-1))
dot = torch_grad * hand_grad / np.expand_dims(torch_norm * hand_norm, axis=-1)
dot = np.sum(dot, axis=-1)
print('Cosine similarity and norm ratio between pytorch and exact computations:')
print(dot)

# Print ratio of magnitudes
print(torch_norm / hand_norm)
print('\n')

### Grad Approximation

# Arbitrary parameters as set in UMAP
n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
n_iters = max(5, int(round(np.log2(X.shape[0]))))
n_neighbors = 15
nn_dict = NNDescent(
    X,
    n_neighbors=n_neighbors,
    n_trees=n_trees,
    n_iters=n_iters,
    max_candidates=20,
    verbose=True
).neighbor_graph

knn_dists = nn_dict['_knn_dists'].astype(np.float32)
knn_indices = nn_dict['_knn_indices']
rows = np.zeros(knn_indices.size, dtype=np.int32)
cols = np.zeros(knn_indices.size, dtype=np.int32)
vals = np.zeros(knn_indices.size, dtype=np.float32)
n_edges = int(rows.shape[0])
for i in range(len(knn_dists)):
    for j in range(n_neighbors):
        val = np.exp(-1 * knn_dists[i, j])
        rows[i * n_neighbors + j] = i
        cols[i * n_neighbors + j] = knn_indices[i, j]
        vals[i * n_neighbors + j] = val
vals /= np.sum(vals)

attr_grads = np.zeros([n_vertices, 2], dtype=np.float32)
rep_grads = np.zeros([n_vertices, 2], dtype=np.float32)
q_ij = np.zeros([n_edges], dtype=np.float32)
q_ik = np.zeros([n_edges], dtype=np.float32)
rand_indices = np.zeros([n_edges], dtype=np.int32)
sym_attraction = True

Y = Y.detach().numpy()
def kernel_function(dist):
    return 1 / (1 + dist)
def sq_euc_dist(y1, y2):
    return np.sum(np.square(y1 - y2))

for edge in range(n_edges):
    j = rows[edge]
    k = cols[edge]
    y1 = Y[j]
    y2 = Y[k]
    dist = sq_euc_dist(y1, y2)
    q_ij[edge] = kernel_function(dist)

    k = np.random.randint(0, n_vertices)
    y2 = Y[k]
    dist = sq_euc_dist(y1, y2)
    q_ik[edge] = kernel_function(dist)
    rand_indices[edge] = k

approx_Z = np.sum(q_ik)
approx_Z *= n_vertices
q_ij /= approx_Z
q_ik /= approx_Z
sum_pq = np.sum(vals * q_ij)
sum_qq = np.sum(q_ik * q_ik) * n_vertices

for edge in range(n_edges):
    j = rows[edge]
    k = cols[edge]
    y1 = Y[j]
    y2 = Y[k]

    q_ik_sq = q_ik[edge] * q_ik[edge]
    attr_force = vals[edge] * q_ij[edge] * q_ij[edge] - sum_pq * q_ik_sq
    grad = attr_force * (y1 - y2)
    attr_grads[j] -= grad
    if sym_attraction:
        attr_grads[k] += grad

    y2 = Y[rand_indices[edge]]
    rep_force = (sum_qq - q_ik[edge]) * q_ik_sq
    grad = rep_force * (y1 - y2)
    rep_grads[j] += grad

approx_grad = 8 * (attr_grads + rep_grads) * approx_Z

attr_norm = np.sqrt(np.sum(np.square(attr_grads), axis=-1))
rep_norm = np.sqrt(np.sum(np.square(rep_grads), axis=-1))
print('Cosine similarity and norm ratio between pytorch and grad approximation:')

torch_norm = np.sqrt(np.sum(np.square(torch_grad), axis=-1))
approx_norm = np.sqrt(np.sum(np.square(approx_grad), axis=-1))
dot = torch_grad * approx_grad / np.expand_dims(torch_norm * approx_norm, axis=-1)
dot = np.sum(dot, axis=-1)
print(dot)
print(torch_norm / approx_norm)
