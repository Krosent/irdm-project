from main import DataReader
import numpy as np
from scipy.sparse import linalg
from scipy.stats import ortho_group
from random import seed
from random import randint


reader = DataReader()
#  matrix_a = reader.sparse_matrix_a().astype(np.float64).tolil()
matrix_b = reader.sparse_matrix_b().astype(np.float64).tolil()
#matrix_b.resize(2000, 1000)
epochs = 3500

# SVD (Initializing P & Q)
U, eps_diag, vT = linalg.svds(matrix_b)
eps = np.diag(eps_diag)
Q = U
pT = np.matmul(eps, vT)
P = pT.T


# Latent Factors
def latent_factors(matrix, q, p):
    R = matrix
    M, N = R.shape

    print("Before: ", len(R.nonzero()[0]))
    for i in range(1, M): #M
        zero_idx = np.where(R[i].toarray()[0] == 0)
        for j in range(len(zero_idx[0])):
            qi = q[i]
            px = p[zero_idx[0][j]]
            rxi = 0
            res = np.dot(qi, px)
            rxi = np.sum(res)
            R[i,zero_idx[0][j]] = round(rxi)
    print("After: ", len(R.nonzero()[0]))
    return R


# Stochastic Gradient Descent
def sgd(matrix, p, q, lam, learning_rate, epochs):
    non_zero_row, non_zero_col = matrix.nonzero()
    for k in range(epochs):
        qG = []
        pG = []
        random = randint(0, len(non_zero_row) - 1)
        x = non_zero_row[random]
        i = non_zero_col[random]
        rxi = matrix[x, i]
        for j in range(len(p[i])):
            estimate = np.matmul(q[x], p[i])
            qG.append((-2 * (rxi - estimate) * p[i][j]) + (2 * lam * q[x][j]))
            pG.append((-2 * (rxi - estimate) * q[x][j]) + (2 * lam * p[i][j]))
        p[i] = p[i] - np.multiply(learning_rate, np.array(pG))
        q[x] = q[x] - np.multiply(learning_rate, np.array(qG))
    return p, q

# Batch Gradient Descent WIP
def bgd(matrix, p, q, lam, gradient_step, epochs):
    non_zero_row, non_zero_col = matrix.nonzero()
    for k in range(epochs):
        gQ = 0
        gP = 0
        for j in range(1, 6): 
            for i in range(len(non_zero_row)):
                row = non_zero_row[i] #idx
                col = non_zero_col[i] #idx
                rui = matrix[row, col]
                estimate = np.matmul(q[row].T, p[col])
                qif = -2 * (rui - estimate) * p[col][j] + (2 * lam * q[row][j])
                pif = -2 * (rui - estimate) * q[row][j] + (2 * lam * q[row][j])
            gQ += qif
            gP += pif
        q = q - np.multiply(gradient_step, gQ)
        p = p - np.multiply(gradient_step * gP)
    return p, q


def accuracy_validation(p, q):
    split_ratio = 0.3
    print(split_ratio)
    num_cols = int(matrix_b.shape[1])
    print(num_cols)
    ## training set
    indx = int(num_cols * split_ratio)
    print(indx)
    training_set = matrix_b[:, :indx]
    test_set = matrix_b[:, indx:]
    # print(training_set)
    print(training_set.shape)
    print(test_set.shape)

    training_p = p[:indx, :]
    test_p = p[indx:, :]

    p_training, q_training = sgd(training_set, training_p, q, 0.5, 0.0001, 10000)
    p_test, q_test = sgd(test_set, test_p, q, 0.5, 0.0001, 10000)

    mse_training = np.square(training_set - q_training.dot(p_training.T)).mean()
    mse_test = np.square(test_set - q_test.dot(p_test.T)).mean()

    print("MSE For Training Set: " + str(mse_training))
    print("MSE For Test Set: " + str(mse_test))

    return 0

if __name__ == '__main__':
    #P, Q = sgd(matrix_b, P, Q, 0.5, 0.1, 15)
    # P, Q = sgd(matrix_b, P, Q, 0.5, 0.0001, 10000)  # result of 3rd task (on matrix B)

    accuracy_validation(P, Q)
    # print(str(training_set))
    ## testing set


    # print(str(P))
    # print(str(Q))



   
        


