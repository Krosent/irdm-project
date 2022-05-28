from collections import defaultdict
from itertools import combinations

from scipy.sparse import csr_matrix, vstack

from main import DataReader
import numpy as np
import random
import time

reader = DataReader()
matrix_a = reader.sparse_matrix_a()
matrix_b = matrix_a.transpose()
norms = np.sqrt(matrix_b.multiply(matrix_b).sum(1))
threshold = 0.5
gamma = 4 * np.log(len(norms)) / threshold

# calculating the norms beforehand, this will save time in the iterations.
# matrix_b.resize(4500, 4500)
# for i in range(4500):
#     norms.append(norm(matrix_b.getrow(i)))
# print(norms)


def map(row):
    result = []

    if row.count_nonzero() >= 1:
        ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
        indices = np.nonzero(row)[1]
        combined = []  # making tuples: (index, rating) for each rating (made tuples so the index and rating is easily accessable inside of the iteration below)
        for i in range(0, len(ratings)):
            combined.append((indices[i], ratings[i]))

        pairs = list(combinations(combined, 2))  # creating pairs between every tuple in combined
        for ai in pairs:  # iterate over all the pairs
            first_index = ai[0][0]  # aij
            first_rating = ai[0][1]
            second_index = ai[1][0]  # aik
            second_rating = ai[1][1]

            cj = first_index
            ck = second_index
            cj_normalized = norms[cj]  # list starts from 0
            ck_normalized = norms[ck]
            #  if cj_normalized != 0.0 and ck_normalized != 0.0:
            formula = gamma * (1 / (cj_normalized * ck_normalized))
            probability = min(1.0, gamma * formula)
            # print("Probability: ", probability)
            rand = random.random()
            if rand <= probability:
                result.append((str(cj) + "-" + str(ck), first_rating * second_rating))
    return result


def reduce(mapper_results):
    # https://docs.python.org/3/library/collections.html#collections.defaultdict
    _dict = defaultdict(list)

    for k, v in mapper_results:
        _key = k.split('-')
        _key = str(_key[1]) + '-' + str(_key[0])
        if _key not in _dict:
            _dict[k].append(float(v))
        else:
            _dict[_key].append(float(v))

    for key, values in _dict.items():
        rating_indexes = key.split("-")

        cj_normalized = norms[int(rating_indexes[0])]  # list starts from 0
        ck_normalized = norms[int(rating_indexes[1])]

        if (gamma / (cj_normalized * ck_normalized)) > 1:
            fst = 1 / (cj_normalized * ck_normalized)
            snd = sum(values)
            return key, fst * snd
        else:
            fst = 1 / gamma
            snd = sum(values)
            return key, fst * snd


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
    # matrix.resize(4500, 4500)
    return matrix


if __name__ == '__main__':
    print("Start Execution")
    start_time = time.time()

    # For building sparse matrix from map-reduce operation
    approximated_rows = []
    approximated_cols = []
    approximated_values = []

    exact_rows = []

    print("--- step 1: Dimsium Application ---")
    for i in range(0, len(reader.users) - 1):
        mapper_result = map(matrix_a.getrow(i))
        # do not apply reduce operation on empty map results
        if mapper_result:
            reducer_result = reduce(mapper_result)
            _row, _col, _value = converter(reducer_result)
            approximated_rows.append(_row)
            approximated_cols.append(_col)
            approximated_values.append(_value)


    print('--- step 2: Dimsium Results Conversion ---')
    #  dimsium results to sparse matrix conversion
    approximated_operation = to_sparse_matrix(approximated_rows, approximated_cols, approximated_values)
    approximated_operation.resize(149, 149)

    print('--- step 3: Exact Operation Calculation ---')
    #  calculation of A^T * T
    exact_operation = matrix_a.transpose().dot(matrix_a)

    print('--- step 3: MSE Calculation ---')
    mse = (np.square(approximated_operation - exact_operation)).mean()
    print("MSE Value: " + str(mse))

    print("End Execution")
    end_time = time.time()
    print("--- %s seconds time execution of the whole task ---" % (end_time - start_time))
