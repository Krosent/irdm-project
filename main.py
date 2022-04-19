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
            limit = limit + 1
            if limit <= 150000:
                movie_id = self.parse_movie_id(line)
                self.parse_user_and_rating(line, movie_id)

    def parse_movie_id(self, line):
        if ':' in line:
            # todo: implement
            movie_id, _ = line.split(':')
            self.latest_movie_id = movie_id
        return self.latest_movie_id

    def parse_user_and_rating(self, line, movie_id):
        if ',' in line:
            user_id, rating, _ = line.split(',')
            # print('body: ' + user_id)
            self.populate_lists(movie_id, user_id, rating)

    def populate_lists(self, movie_id, user_id, rating):
        self.movies.append(movie_id)
        self.users.append(user_id)
        self.ratings.append(rating)


if __name__ == '__main__':
    reader = DataReader()
    print('Test Inputs')
    print('Movie ids: ' + str(reader.movies[:10]))
    print('User ids: ' + str(reader.users[:10]))
    print('Rating ids: ' + str(reader.ratings[:10]))
