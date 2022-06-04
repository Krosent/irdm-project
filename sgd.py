from main import DataReader
import numpy as np
from scipy.sparse import linalg
from scipy.stats import ortho_group
from random import seed
from random import randint
import copy
import matplotlib.pyplot as plt 
import pandas as pd

reader = DataReader()
matrix_b = reader.sparse_matrix_b().astype(np.float64).tolil()
matrix_b.resize(149, 100000)

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
def sgd(matrix, p, q, lam, learning_rate, epochs, mse_every_epoch):
    # make a deep copy of objects
    _p = copy.deepcopy(p)
    _q = copy.deepcopy(q)
    accuracy_history = []
    non_zero_row, non_zero_col = matrix.nonzero()
    for k in range(epochs):
        qG = []
        pG = []
        random = randint(0, len(non_zero_row) - 1)
        x = non_zero_row[random]
        i = non_zero_col[random]
        rxi = matrix[x, i]
        for j in range(len(_p[i])):
            estimate = np.matmul(_q[x], _p[i])
            qG.append((-2 * (rxi - estimate) * _p[i][j]) + (2 * lam * _q[x][j]))
            pG.append((-2 * (rxi - estimate) * _q[x][j]) + (2 * lam * _p[i][j]))
        _p[i] = _p[i] - np.multiply(learning_rate, np.array(pG))
        _q[x] = _q[x] - np.multiply(learning_rate, np.array(qG))

        # calculate MSE for each 1k epoch and add it to the history
        if k % mse_every_epoch == 0:
            mse = np.square(matrix - _q.dot(_p.T)).mean()
            accuracy_history.append(mse)
    return _p, _q, accuracy_history

# Batch Gradient Descent WIP
def bgd(matrix, p, q, lam, learning_rate, epochs, mse_every_epoch):
    # make a deep copy of objects
    _p = copy.deepcopy(p)
    _q = copy.deepcopy(q)
    accuracy_history = []
    non_zero_row, non_zero_col = matrix.nonzero()
    estimates = []
    for k in range(epochs):
        gQ = np.zeros(q.shape)
        gP = np.zeros(p.shape)
        # Loop through all know rating every iteration to create a gradient matrix for P and Q
        print("EPOCH: ", k)
        print("q[first_non_zero]: ", _q[non_zero_row[1]], " p[first_non_zero]: ", _p[non_zero_col[1]])
        for r in range(len(non_zero_row)):
            qx_g = [] # gradient for row x in q
            pi_g = [] # gradient for row i in p
            x = non_zero_row[r]
            i = non_zero_col[r]
            rxi = matrix[x, i]
            #print("Q: ", _q.shape, " P: ", _p.shape)
            estimate = np.matmul(_q[x], _p[i])
            estimates.append(np.matmul(_q[x], _p[i]))
            for j in range(len(_p[i])):
                qx_g.append((-2 * (rxi - estimate) * _p[i][j]) + (2 * lam * _q[x][j]))
                pi_g.append((-2 * (rxi - estimate) * _q[x][j]) + (2 * lam * _p[i][j]))
            # update the row in the gradient matrices
            gP[i] += pi_g
            gQ[x] += qx_g
        print("Max estimate: ", max(estimates))
        print("====================")
        # update p and q after looping through all know ratings
        _p = _p - np.multiply(learning_rate, gP)
        _q = _q - np.multiply(learning_rate, gQ)

        if k % mse_every_epoch == 0:
            mse = np.square(matrix - _q.dot(_p.T)).mean()
            accuracy_history.append(mse)
    return _p, _q, accuracy_history


def accuracy_validation(p, q, epochs, mse_every_epoch):
    split_ratio = 0.3
    num_rows, num_cols = matrix_b.shape
    indx_p = int(num_cols * split_ratio)
    indx_q = int(num_rows * split_ratio)

    training_set = matrix_b[:, :indx_p]
    test_set = matrix_b[:, indx_p:]

    training_p = p[:indx_p, :]
    test_p = p[indx_p:, :]

    training_q = q[:, :indx_q]
    test_q = q[:, indx_q:]

    p_training, q_training, accuracy_history_training = sgd(training_set, training_p, q, 0.5, 0.000005, epochs, mse_every_epoch)
    p_test, q_test, accuracy_history_test = sgd(test_set, test_p, q, 0.5, 0.000005, epochs, mse_every_epoch)

    mse_training = np.square(training_set - q_training.dot(p_training.T)).mean()
    mse_test = np.square(test_set - q_test.dot(p_test.T)).mean()


    fig1, ax1 = plt.subplots()
    fig1.suptitle('MSE History - Epochs: ' + str(epochs))
    ax1.plot(accuracy_history_training)
    ax1.set_title("Training Set")
    ax1.set_xticks(range(0, len(accuracy_history_training), 10))
    ax1.set_xticklabels(range(0, epochs, 250))

    fig2, ax2 = plt.subplots()
    fig2.suptitle('MSE History - Epochs: ' + str(epochs))
    ax2.plot(accuracy_history_test)
    ax2.set_title("Test set")
    ax2.set_xticks(range(0, len(accuracy_history_test), 10))
    ax2.set_xticklabels(range(0, epochs, 250))
    plt.show()
    print("accuracy history training: ", accuracy_history_training)
    print("MSE For Training Set: " + str(mse_training))
    #print("MSE For Test Set: " + str(mse_test))
    return 0


epochs = 2500
mse_every_epoch = 25
if __name__ == '__main__':
    _P_validation = copy.deepcopy(P)
    _Q_validation = copy.deepcopy(Q)
    accuracy_validation(_P_validation, _Q_validation, epochs, mse_every_epoch)

   # P, Q, history = sgd(matrix_b, P, Q, 0.5, 0.0001, 10000)  # result of 3rd task (on matrix B)
