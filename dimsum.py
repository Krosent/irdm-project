from collections import defaultdict
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
    result = []
    if row.count_nonzero() > 1:
        ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
        indices = np.nonzero(row)[1]
        combined = []  # making tuples: (index, rating) for each rating (made tuples so the index and rating is easily accessable inside of the iteration below)
        for i in range(0, len(ratings)):
            combined.append((indices[i], ratings[i]))

        pairs = list(combinations(combined, 2))  # creating pairs between every tuple in combined
        result = []
        for ai in pairs:  # iterate over all the pairs
            first_index = ai[0][0]  # aij
            first_rating = ai[0][1]
            second_index = ai[1][0]  # aik
            second_rating = ai[1][1]

            cj = first_index
            ck = second_index
            cj_normalized = norms[cj - 1]  # list starts from 0
            ck_normalized = norms[ck - 1]
            if cj_normalized != 0.0 and ck_normalized != 0.0:
                formula = gamma * (1 / (cj_normalized * ck_normalized))
                probability = min(1, gamma * formula)
                # print("Probability: ", probability)
                rand = random.uniform(0, 1)
                if rand <= probability:
                    result.append((str(cj) + "-" + str(ck), first_rating * second_rating))
    return result


def reduce(mapper_results, gamma):
    # https://docs.python.org/3/library/collections.html#collections.defaultdict
    _dict = defaultdict(list)

    for k, v in mapper_results:
        _dict[k].append(float(v))

    for key, values in _dict.items():
        rating_indexes = key.split("-")

        if (gamma / (float(rating_indexes[0]) * float(rating_indexes[1]))) < 1:
            fst = 1 / (float(rating_indexes[0]) * float(rating_indexes[1]))
            snd = sum(values)

            return key, fst * snd
        else:
            fst = 1 / gamma
            snd = sum(values)

            return key, fst * snd


if __name__ == '__main__':
    for i in range(0, 100):
        gamma = 0.4
        # approximation
        mapper_result = map(matrix_a.getrow(i), gamma)
        if mapper_result:
            # print("row: " + str(matrix_a.getrow(i)))
            reducer_result = reduce(mapper_result, gamma)
            print("dimsum results: " + str(reducer_result))
            # print("---------")
            # print(str((matrix_a.getrow(i).transpose()).multiply(matrix_a.getrow(i))))
            # print("xxxxxxxx")
