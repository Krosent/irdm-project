# import statements here:
# todo

# define empty list for our data
movies = []
users = []
ratings = []
curr_movie_id = 0


def parse_input():
    input_1 = open('netflix_dataset/combined_data_1.txt', 'r')
    input_1_lines = input_1.readlines()

    limit = 0
    for line in input_1_lines:
        limit = limit + 1
        if limit <= 150000:
            movie_id = parse_movie_id(line)
            parse_user_and_rating(line, movie_id)


def parse_movie_id(line):
    global curr_movie_id
    if ':' in line:
        # todo: implement
        movie_id, _ = line.split(':')
        curr_movie_id = movie_id
    return curr_movie_id


def parse_user_and_rating(line, movie_id):
    if ',' in line:
        # todo: implement
        user_id, rating, _ = line.split(',')
        # print('body: ' + user_id)
        populate_lists(movie_id, user_id, rating)


def populate_lists(movie_id, user_id, rating):
    movies.append(movie_id)
    users.append(user_id)
    ratings.append(rating)


if __name__ == '__main__':
    parse_input()
    print('Test Inputs')
    print('Movie ids: ' + str(movies[:10]))
    print('User ids: ' + str(users[:10]))
    print('Rating ids: ' + str(ratings[:10]))
