import torch
import numpy as np

X = torch.normal(0, 1, [100, 50])
Y = torch.normal(0, 1, [100, 2], requires_grad=True)

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
print(dot)

# Print ratio of magnitudes
print(torch_norm / hand_norm)
