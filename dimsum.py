from itertools import combinations
from main import DataReader
import numpy as np
from scipy.sparse.linalg import norm
import random

reader = DataReader()
matrix_a = reader.sparse_matrix_a()
matrix_b = reader.sparse_matrix_b()
norms = []


# calculating the norms beforehand, this will save time in the iterations.

for i in range(2649429):
    norms.append(norm(matrix_a.getrow(i)))

#print(norms)
#print(len(norms))


def map(row, gamma):
    if row.count_nonzero() > 1:
        ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
        indices = np.nonzero(row)[1]
        combined = []  # making tuples: (index, rating) for each rating (made tuples so the index and rating is easily accessable inside of the iteration below)
        for i in range(0, len(ratings)):
            combined.append((indices[i], ratings[i]))

        pairs = list(combinations(combined, 2))  # creating pairs between every tuple in combined
        for a_ij in pairs:  # iterate over all the pairs
            # print("pairs" + str(len(pairs)))
            # print("aij" + str(a_ij))
            first_index = a_ij[0][0]
            first_rating = a_ij[0][1]
            second_index = a_ij[1][0]
            second_rating = a_ij[1][1]

            probability = 0
            # norm can be zero and we can't allow division on zero, so we have checks here
            if norms[first_index - 1] != 0 and norms[second_index - 1] != 0:
                probability = min(1, gamma * (1 / (norms[first_index - 1] * norms[second_index - 1])))

            print("Probability: ", probability)

            if random.uniform(0, 1) <= probability:
                print("EMIT")  # TODO


def reduce():
    return "reduce"


# Iteration for 10 rows
if __name__ == '__main__':
    for i in range(0, 10):
        map(matrix_a.getrow(i), 0.5)
