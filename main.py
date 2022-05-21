import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csr_array, coo_matrix


class DataReader:
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
            #limit = limit + 1
            #if limit <= 1_000_000:
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

        # users are rows, movies are columns

    def sparse_matrix_a(self):
        return csr_matrix((np.array(self.ratings, dtype=np.int64),
                           (np.array(self.users, dtype=np.int64),
                            np.array(self.movies, dtype=np.int64))))

    # movies are rows, users are columns
    def sparse_matrix_b(self):
        return csr_matrix((np.array(self.ratings, dtype=np.int64),
                          (np.array(self.movies, dtype=np.int64),
                           np.array(self.users, dtype=np.int64))))


if __name__ == '__main__':
    reader = DataReader()
    # print('Test Inputs')
    # print('Movie ids: ' + str(reader.movies[:10]))
    # print('User ids: ' + str(reader.users[:10]))
    # print('Rating ids: ' + str(reader.ratings[:10]))
    # print('min ' + str(min(reader.movies)))
    # print('min ' + str(min(reader.users)))
    matrix_a = reader.sparse_matrix_a()
    matrix_b = reader.sparse_matrix_b()

    print("---- MATRIX A ----")
    print(matrix_a)
    print("----  ----")
    print("---- MATRIX B ----")
    print(matrix_b)
    print("----  ----")
