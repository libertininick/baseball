import numpy as np

a = np.array([[3, -2, 1]
            , [1, 1, 1]
              ,[3, -2, -1]])

b = np.array([7
            , 2
              , 3])

print(np.linalg.solve(a, b))

print(np.dot(np.linalg.inv(a), b))

v = np.array([-4, -3, 8])
b1 = np.array([1, 2, 3])
b2 = np.array([-2, 1, 0])
b3 = np.array([-3, -6, 5])


# Check that b's are orthogonal
print((np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))) == 0)
print((np.dot(b1, b3) / (np.linalg.norm(b1) * np.linalg.norm(b3))) == 0)
print((np.dot(b2, b3) / (np.linalg.norm(b2) * np.linalg.norm(b3))) == 0)

# Scalar projections
v_b1_scalar_projection = np.dot(v, b1) / np.dot(b1, b1)
v_b2_scalar_projection = np.dot(v, b2) / np.dot(b2, b2)
v_b3_scalar_projection = np.dot(v, b3) / np.dot(b3, b3)

v_b = np.array([v_b1_scalar_projection, v_b2_scalar_projection, v_b3_scalar_projection])
print(v_b)

a = np.array([[1, 2, -1]])
b = np.array([[3, -4, 5]])
c = np.array([[1, -8, 7]])

m = np.hstack((a.T, b.T, c.T))

np.linalg.det(m)

v = np.array([2, 1])
b1 = np.array([3, -4])

np.dot(v, b1)
np.dot(b1, b1)

np.hstack
np.sqrt((6/25)**2 + (8/25)**2)


A = np.random.randint(1, 10, 3*4).reshape((3, 4))
B = np.random.randint(1, 10, 4*4).reshape((4, 4))

C = np.empty((A.shape[0], B.shape[1]))
for i in range(A.shape[0]):
    for k in range(B.shape[1]):
        c_ik = 0
        for j in range(A.shape[1]):
            c_ik += A[i, j]*B[j, k]

        C[i, k] = c_ik

assert (np.matmul(A, B) == C).all()


T = np.array([[1, 0], [2, -1]])
C = np.array([[1, 0], [1, 1]])


print(np.round(np.linalg.inv(C) @ T @ C, 2))

np.array([[1,0],[1,1]]) @ (np.array([[1,0],[0,-1]])**5) @ np.array([[1, 0], [-1, 1]])

A = np.array([[3/2, -1], [-1/2, 1/2]])
eVals, eVecs = np.linalg.eig(A)

1 - 3**0.5/2
1 + 3**0.5/2

C = np.hstack(((eVecs[:, 0] / eVecs[1, 0]).reshape(2,1)
               , (eVecs[:, 1] / eVecs[1, 1]).reshape(2,1)))

D = np.round(np.linalg.inv(C) @ A @ C, 20)

C @ D**2 @ np.linalg.inv(C)


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


n, m = (3, 5000)
X = np.random.randn(n*m).reshape((n, -1))
w_actual = np.array([0.75, -0.5, 0.15]).reshape((3, 1))
b_actual = -0.2
y = np.where(w_actual.T @ X + b_actual + np.random.randn(m) > 0, 1, 0)
print(y.mean())

w_learn = (np.random.randn(n) / 100).reshape((3, 1))
b_learn = np.random.randn(1) / 100

alpha = 0.05

for epoc in range(1000):

    Z = w_learn.T @ X + b_learn
    A = sigmoid(Z)
    C = (-1/m)*np.sum(y*np.log(A) + (1 - y)*np.log(1 - A))

    dCdA = (-1/m)*(y/A - (1 - y)/(1 - A))
    dAdZ = sigmoid(Z)*(1 - sigmoid(Z))
    dZdw = X
    dZdb = 1

    dCdw = dZdw @ (dCdA*dAdZ).T
    dCdb = dZdb*np.sum(dCdA*dAdZ)

    w_learn -= alpha*dCdw
    b_learn -= alpha*dCdb

    if (epoc + 1) % 100 == 0 or epoc == 0:
        print(f'Epoc {epoc+1}: Cost = {round(C,3)}; Parms = {np.append(b_learn, w_learn)}')

x1, x2, x3 = w_learn
print(b_learn, x1, x2, x3)

import statsmodels.api as sm
logit_model = sm.Logit(y.T, sm.add_constant(X.T))
result = logit_model.fit(method='newton', maxiter=300)
print(result.summary2())