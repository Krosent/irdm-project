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
# I assume that we need to take random 4500 columns for each row
matrix_b.resize(4500, 4500)
for i in range(4500):
    norms.append(norm(matrix_b.getrow(i)))
print(norms)


def map(row, gamma):
    if row.count_nonzero() > 1:
        ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
        indices = np.nonzero(row)[1]
        combined = []  # making tuples: (index, rating) for each rating (made tuples so the index and rating is easily accessable inside of the iteration below)
        for i in range(0, len(ratings)):
            combined.append((indices[i], ratings[i]))

        pairs = list(combinations(combined, 2))  # creating pairs between every tuple in combined
        for ai in pairs:  # iterate over all the pairs
            first_index = ai[0][0]
            first_rating = ai[0][1]
            second_index = ai[1][0]
            second_rating = ai[1][1]

            aij = norms[first_index - 1]
            aik = norms[second_index - 1]
            if aij != 0.0 and aik != 0.0:
                formula = gamma * (1 / (aij * aik))
                probability = min(1, gamma * formula)
                print("Probability: ", probability)
                rand = random.uniform(0, 1)
                if rand <= probability:
                    print("EMIT")  # TODO

def reduce():
    return "reduce"


if __name__ == '__main__':
    print("shape: " + str(matrix_b.shape))
    for i in range(0, 10):
        map(matrix_a.getrow(i), 0.8)
