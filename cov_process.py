#!/usr/bin/env python
import numpy as np

# find E, so that A^-1 = E^T * E
def MatrixInvSqrt(A):
    eigvalues, eigvectors = np.linalg.eig(A)
    A_inv_sqrt = np.matmul(eigvectors, np.diag(1./np.sqrt(eigvalues)))
    return A_inv_sqrt.transpose()

    

print("B0----------")
B0 = np.array([[1.2,0.6], [0.6,0.6]], np.float)
B0_inv_sqrt = MatrixInvSqrt(B0)
print(B0_inv_sqrt)
print(np.matmul(B0_inv_sqrt.transpose(), B0_inv_sqrt))

print("Ra-----------")
Ra = np.array([[0.6,-0.6],[-0.6,1.2]], np.float)
Ra_inv_sqrt = MatrixInvSqrt(Ra)
print(Ra_inv_sqrt)
print(np.matmul(Ra_inv_sqrt.transpose(), Ra_inv_sqrt))

print("Rb-----------")
Rb = np.array([[0.6,0.6],[0.6,1.2]], np.float)
Rb_inv_sqrt = MatrixInvSqrt(Rb)
print(Rb_inv_sqrt)
print(np.matmul(Rb_inv_sqrt.transpose(), Rb_inv_sqrt))


print("inv(B0) + inv(Ra) * 7 + inv(Rb) * 7")
CovX = np.linalg.inv(np.linalg.inv(B0) + np.linalg.inv(Ra) * 7 + np.linalg.inv(Rb)*7)
print(CovX)

