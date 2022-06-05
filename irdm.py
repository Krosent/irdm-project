import numpy as np
from scipy.sparse import csr_matrix, linalg
from collections import defaultdict
from itertools import combinations
import random
import time
from sklearn.preprocessing import normalize
from random import randint
import copy
import matplotlib.pyplot as plt

# The code is divided into logical steps (classes) where each class is responsible for certain task
# The authors: Vandevivere Achiel and Kuznetsov Artyom


class DataReader:
    """
    The class is responsible for reading data from txt file and parsing it into sparse matrices.
    We decided to read only portion of the file, since it is quite big. You can tweak the number for your experiments.
    Our code expect to store dataset in 'netflix_dataset' folder.
    Please modify the following line in case you want to change input path:
    **  input_1 = open('netflix_dataset/combined_data_1.txt', 'r')
    """

    movies = []
    users = []
    ratings = []
    latest_movie_id = 0

    def __init__(self):
        self.parse_input()

    def parse_input(self):
        input_1 = open('netflix_dataset/combined_data_1.txt', 'r')
        input_1_lines = input_1.readlines()

        limit = 0
        for line in input_1_lines:
            limit = limit + 1
            if limit <= 500_000:
                movie_id = self.parse_movie_id(line)
                self.parse_user_and_rating(line, movie_id)

    def parse_movie_id(self, line):
        if ':' in line:
            self.latest_movie_id = line.split(':')[0]
        return self.latest_movie_id

    def parse_user_and_rating(self, line, movie_id):
        if ',' in line:
            user_id, rating, _ = line.split(',')
            self.populate_lists(movie_id, user_id, rating)

    def populate_lists(self, movie_id, user_id, rating):
        self.movies.append(movie_id)
        self.users.append(user_id)
        self.ratings.append(rating)

    def sparse_matrix_a(self):
        return csr_matrix((np.array(self.ratings, dtype=np.int64),
                           (np.array(self.users, dtype=np.int64),
                            np.array(self.movies, dtype=np.int64))))

    def sparse_matrix_b(self):
        return csr_matrix((np.array(self.ratings, dtype=np.int64),
                          (np.array(self.movies, dtype=np.int64),
                           np.array(self.users, dtype=np.int64))))


class Dimsum:
    """
    This class is responsible for Dimsium Algorithm implementation.
    We use DataReader class to read data from the input path.
    Our algorithm is seperated into map, reduce methods that do most of the jobs.
    The rest of the functions inside are util functions.
    """

    reader = DataReader()
    matrix_a = reader.sparse_matrix_a()
    matrix_b = matrix_a.transpose()
    norms = np.sqrt(matrix_b.multiply(matrix_b).sum(1))
    threshold = 0.1
    gamma = 4 * np.log(len(norms)) / threshold

    def map(self, row):
        result = []

        if row.count_nonzero() >= 1:
            ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
            indices = np.nonzero(row)[1]
            # making tuples: (index, rating) for each rating
            # (made tuples so the index and rating is easily accessible inside of the iteration below)
            combined = []
            for _iter in range(0, len(ratings)):
                combined.append((indices[_iter], ratings[_iter]))

            pairs = list(combinations(combined, 2))  # creating pairs between every tuple in combined
            for ai in pairs:  # iterate over all the pairs
                first_index, first_rating = ai[0]  # aij
                second_index, second_rating = ai[1]  # aik

                cj = first_index
                ck = second_index
                cj_normalized = self.norms[cj]  # list starts from 0
                ck_normalized = self.norms[ck]
                formula = self.gamma * (1 / (cj_normalized * ck_normalized))
                probability = min(1.0, self.gamma * formula)
                rand = random.uniform(0, 1)
                if rand <= probability:
                    result.append((str(cj) + "-" + str(ck), first_rating * second_rating))
        return result

    def reduce(self, mapper_results):
        _dict = self.prepare_reduce_input(mapper_results)

        for key, values in _dict.items():
            rating_indexes = key.split("-")

            cj_normalized = self.norms[int(rating_indexes[0])]
            ck_normalized = self.norms[int(rating_indexes[1])]

            if (self.gamma / (cj_normalized * ck_normalized)) > 1:
                fst = 1 / (cj_normalized * ck_normalized)
                snd = sum(values)
                return key, fst * snd
            else:
                fst = 1 / self.gamma
                snd = sum(values)
                return key, fst * snd

    """
    Preparation function that simulates reduce operation in pure python
    """
    def prepare_reduce_input(self, mapper_results):
        # https://docs.python.org/3/library/collections.html#collections.defaultdict
        _dict = defaultdict(list)
        for k, v in mapper_results:
            _key = k.split('-')
            _key = str(_key[1]) + '-' + str(_key[0])
            if _key not in _dict:
                _dict[k].append(float(v))
            else:
                _dict[_key].append(float(v))
        return _dict

    def converter(self, result):
        if result is not None:
            row = result[0].split("-")[0]
            col = result[0].split("-")[1]
            value = result[1]
            return row, col, value

    def to_sparse_matrix(self, rows, cols, values):
        matrix = csr_matrix((np.array(values, dtype=np.float64),
                             (np.array(rows, dtype=np.int64),
                              np.array(cols, dtype=np.int64))))
        return matrix


class GradientDescent:
    # Function returns Latent Factors
    def latent_factors(self, matrix, q, p):
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
    def sgd(self, matrix, p, q, lam, learning_rate, epochs, mse_every_epoch):
        """
        Making a deepcopy of p, q in order to avoid reference pointing and side effects outside the function
        """

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

    # Batch Gradient Descent
    def bgd(self, matrix, p, q, lam, learning_rate, epochs, mse_every_epoch):
        """
        Making a deepcopy of p, q in order to avoid reference pointing and side effects outside the function
        """

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
                qx_g = []  # gradient for row x in q
                pi_g = []  # gradient for row i in p
                x = non_zero_row[r]
                i = non_zero_col[r]
                rxi = matrix[x, i]
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

    def accuracy_validation(self, matrix, p, q, epochs, mse_every_epoch):
        """
        This function is intended to calculate accuracy metrics for Gradient Descent algorithms.
        We calculate MSE as a main metrics to evaluate efficiency of algorithm.
        matplotlib is used for data visualization purposes.
        """
        split_ratio = 0.3
        num_rows, num_cols = matrix.shape
        indx_p = int(num_cols * split_ratio)
        indx_q = int(num_rows * split_ratio)

        training_set = matrix[:, :indx_p]
        test_set = matrix[:, indx_p:]

        training_p = p[:indx_p, :]
        test_p = p[indx_p:, :]

        training_q = q[:, :indx_q]
        test_q = q[:, indx_q:]

        p_training, q_training, accuracy_history_training = self.sgd(training_set, training_p, q, 0.5, 0.000005, epochs, mse_every_epoch)
        p_test, q_test, accuracy_history_test = self.sgd(test_set, test_p, q, 0.5, 0.000005, epochs, mse_every_epoch)

        mse_training = np.square(training_set - q_training.dot(p_training.T)).mean()
        mse_test = np.square(test_set - q_test.dot(p_test.T)).mean()

        fig1, ax1 = plt.subplots()
        fig1.suptitle('MSE History - Epochs: ' + str(epochs))
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE")
        ax1.plot(accuracy_history_training)
        ax1.set_title("Training Set")
        ax1.set_xticks(range(0, len(accuracy_history_training), 100))
        ax1.set_xticklabels(range(0, epochs, 300))

        fig2, ax2 = plt.subplots()
        fig2.suptitle('MSE History - Epochs: ' + str(epochs))
        ax2.plot(accuracy_history_test)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("MSE")
        ax2.set_title("Test set")
        ax2.set_xticks(range(0, len(accuracy_history_test), 100))
        ax2.set_xticklabels(range(0, epochs, 300))
        plt.show()
        print("accuracy history training: ", accuracy_history_training)
        print("MSE For Training Set: " + str(mse_training))
        print("MSE For Test Set: " + str(mse_test))
        return 0

    def evaluate_p_q(self, matrix, k):
        """
        A method that is used to calculate SVD of certain matrix with specific K value
        """
        # SVD (Initializing P & Q)
        U, eps_diag, vT = linalg.svds(A=matrix, k=k)
        eps = np.diag(eps_diag)
        Q = U
        pT = np.matmul(eps, vT)
        P = pT.T
        return P, Q


def run_program(algorithm) :
    if algorithm == "sgd" :
        print("Start SGD Execution")
        epochs = 3000
        mse_every_epoch = 3

        reader = DataReader()
        matrix_b = reader.sparse_matrix_b().astype(np.float64).tolil()
        matrix_b.resize(149, 10000)
    
        gradient_descent = GradientDescent()
        for i in range(1, 15):
            P, Q = gradient_descent.evaluate_p_q(matrix=matrix_b, k=i)

            _P_validation = copy.deepcopy(P)
            _Q_validation = copy.deepcopy(Q)
            print("K=" + str(i))
            gradient_descent.accuracy_validation(matrix=matrix_b, p=_P_validation,
                                                    q=_Q_validation, epochs=epochs,
                                                    mse_every_epoch=mse_every_epoch)
        print("End SGD Execution")
    
    elif algorithm == "dimsum" :
        print("Start DIMSUM Execution")
        dimsum = Dimsum()

        threshold = 0.0
        for execution in range(10):
            print("execution number: " + str(execution))
            if threshold < 0.9:
                threshold = threshold + 0.1
                gamma = 4 * np.log(len(dimsum.norms)) / threshold
            print("threshold: " + str(threshold))
            print("--- step 1: Dimsium Application ---")
            # For building sparse matrix from map-reduce operation
            approximated_rows = []
            approximated_cols = []
            approximated_values = []

            start_time = time.time()
            for i in range(0, len(dimsum.reader.users) - 1):
                mapper_result = dimsum.map(dimsum.matrix_a.getrow(i))
                # do not apply reduce operation on empty map results
                if mapper_result:
                    reducer_result = dimsum.reduce(mapper_result)
                    _row, _col, _value = dimsum.converter(reducer_result)
                    approximated_rows.append(_row)
                    approximated_cols.append(_col)
                    approximated_values.append(_value)
            end_time = time.time()
            print("--- %s seconds dimsium time execution ---" % (end_time - start_time))

            print('--- step 2: Dimsium Results Conversion ---')
            #  dimsium results to sparse matrix conversion
            approximated_operation = dimsum.to_sparse_matrix(approximated_rows, approximated_cols, approximated_values)
            approximated_operation.resize(149, 149)

            print('--- step 3: Exact Operation Calculation ---')
            #  calculation of A^T * T
            exact_operation = dimsum.matrix_a.transpose().dot(dimsum.matrix_a)
            #  normalize the results to be able to calculate MSE
            exact_operation_normed = normalize(exact_operation, axis=0, norm='l1')

            print('--- step 3: MSE Calculation ---')
            mse = (np.square(approximated_operation - exact_operation_normed)).mean()
            print("MSE Value: " + str(mse))
            print("End DIMSUM Execution")

if __name__ == '__main__':
    #run_program("dimsum")
    run_program("sgd")
