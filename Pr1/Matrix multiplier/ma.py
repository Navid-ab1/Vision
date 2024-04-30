from matplotlib import pyplot as plt
import numpy as np
A = np.random.rand(200,10)
mu = np.zeros(A.shape[1])
for i in range(A.shape[0]):
    mu += A[i]
mu /= A.shape[0]
B = np.zeros_like(A)
for i in range(A.shape[0]):
    B[i] = A[i] - mu
print(B)
C = A-np.mean(A,axis=0)
print(C)
