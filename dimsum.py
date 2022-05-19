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
for i in range(1, 31):
    norms.append(norm(matrix_b.getrow(i)))
print(norms)

def map(row, gamma):
    if row.count_nonzero() > 1:
        ratings = np.delete(row.toarray()[0], np.where(row.toarray()[0] == 0))
        indices =  np.nonzero(row)[1]
        combined = [] # making tuples: (index, rating) for each rating (made tuples so the index and rating is easily accessable inside of the iteration below)
        for i in range(0, len(ratings)):
            combined.append((indices[i], ratings[i]))
            
        pairs = list(combinations(combined, 2)) # creating pairs between every tuple in combined
        for i in pairs : # iterate over all the pairs
            first_index = i[0][0]
            first_rating = i[0][1]
            second_index = i[1][0]
            second_rating = i[1][1]
            
            probability = min(1, gamma * (1 / (norms[first_index -1] * norms[second_index -1])))
            print("Probability: ", probability)

            if random.uniform(0,1) <= probability :
               print("EMIT") #TODO

def reduce():
    return "reduce"

# Iteration for 10 rows
for i in range(0,10):
   map(matrix_a.getrow(i), 0.5)
