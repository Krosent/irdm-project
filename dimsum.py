from collections import defaultdict
from itertools import combinations

from scipy.sparse import csr_matrix

from main import DataReader
import numpy as np
import random
import time
from sklearn.preprocessing import normalize

reader = DataReader()
matrix_a = reader.sparse_matrix_a()
matrix_b = matrix_a.transpose()
norms = np.sqrt(matrix_b.multiply(matrix_b).sum(1))
threshold = 0.1
gamma = 4 * np.log(len(norms)) / threshold


def map(row):
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
            cj_normalized = norms[cj]  # list starts from 0
            ck_normalized = norms[ck]
            formula = gamma * (1 / (cj_normalized * ck_normalized))
            probability = min(1.0, gamma * formula)
            rand = random.uniform(0, 1)
            if rand <= probability:
                result.append((str(cj) + "-" + str(ck), first_rating * second_rating))
    return result


def reduce(mapper_results):
    _dict = prepare_reduce_input(mapper_results)

    for key, values in _dict.items():
        rating_indexes = key.split("-")

        cj_normalized = norms[int(rating_indexes[0])]
        ck_normalized = norms[int(rating_indexes[1])]

        if (gamma / (cj_normalized * ck_normalized)) > 1:
            fst = 1 / (cj_normalized * ck_normalized)
            snd = sum(values)
            return key, fst * snd
        else:
            fst = 1 / gamma
            snd = sum(values)
            return key, fst * snd


def prepare_reduce_input(mapper_results):
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


def converter(result):
    if result is not None:
        row = result[0].split("-")[0]
        col = result[0].split("-")[1]
        value = result[1]
        return row, col, value


def to_sparse_matrix(rows, cols, values):
    matrix = csr_matrix((np.array(values, dtype=np.float64),
                         (np.array(rows, dtype=np.int64),
                          np.array(cols, dtype=np.int64))))
    return matrix


if __name__ == '__main__':
    print("Start Execution")

    threshold = 0.0
    for execution in range(10):
        print("execution number: " + str(execution))
        if threshold < 0.9:
            threshold = threshold + 0.1
            gamma = 4 * np.log(len(norms)) / threshold
        print("threshold: " + str(threshold))
        print("--- step 1: Dimsium Application ---")
        # For building sparse matrix from map-reduce operation
        approximated_rows = []
        approximated_cols = []
        approximated_values = []

        start_time = time.time()
        for i in range(0, len(reader.users) - 1):
            mapper_result = map(matrix_a.getrow(i))
            # do not apply reduce operation on empty map results
            if mapper_result:
                reducer_result = reduce(mapper_result)
                _row, _col, _value = converter(reducer_result)
                approximated_rows.append(_row)
                approximated_cols.append(_col)
                approximated_values.append(_value)
        end_time = time.time()
        print("--- %s seconds dimsium time execution ---" % (end_time - start_time))

        print('--- step 2: Dimsium Results Conversion ---')
        #  dimsium results to sparse matrix conversion
        approximated_operation = to_sparse_matrix(approximated_rows, approximated_cols, approximated_values)
        approximated_operation.resize(149, 149)

        print('--- step 3: Exact Operation Calculation ---')
        #  calculation of A^T * T
        exact_operation = matrix_a.transpose().dot(matrix_a)
        #  normalize the results to be able to calculate MSE
        exact_operation_normed = normalize(exact_operation, axis=0, norm='l1')

        print('--- step 3: MSE Calculation ---')
        mse = (np.square(approximated_operation - exact_operation_normed)).mean()
        print("MSE Value: " + str(mse))
        print("End Execution")
