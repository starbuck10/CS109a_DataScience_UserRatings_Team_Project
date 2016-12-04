import time
from datetime import datetime

import numpy as np
import pandas as pd


def date_parse(time_in_secs):
    return datetime.utcfromtimestamp(float(time_in_secs))


def read_ratings_df_with_timestamp(file_name):
    start = time.time()
    ratings_df = pd.read_csv(file_name, parse_dates=['timestamp'], date_parser=date_parse)
    elapsed = time.time() - start
    print 'loaded csv: %.2f secs' % elapsed
    return ratings_df


def read_ratings_df(file_name):
    start = time.time()
    ratings_df = pd.read_csv(file_name)
    elapsed = time.time() - start
    print 'loaded csv: %.2f secs' % elapsed
    return ratings_df


class MovieMatrixData(object):
    def __init__(self, ratings_df):
        start = time.time()

        self.ratings_df = ratings_df
        self.users = np.array(sorted(ratings_df['userId'].unique()))
        self.movies = np.array(sorted(ratings_df['movieId'].unique()))

        user_id_to_index_dict = {user_id: index for index, user_id in enumerate(self.users)}
        movie_id_to_index_dict = {movie_id: index for index, movie_id in enumerate(self.movies)}

        self.rating_matrix = np.zeros((len(self.users), len(self.movies)))

        for _, row in self.ratings_df.iterrows():
            user_index = user_id_to_index_dict[row['userId']]
            movie_index = movie_id_to_index_dict[row['movieId']]
            self.rating_matrix[user_index, movie_index] = row['rating']

        elapsed = time.time() - start
        print 'matrix init: %.2f secs' % elapsed


def explore_rating_matrix(movie_matrix_data):
    rating_matrix = movie_matrix_data.rating_matrix

    nonzero = np.count_nonzero(rating_matrix)

    print 'rating matrix size: {:,d}'.format(rating_matrix.size)
    print 'nonzero elements: {:,d} (density: {:.2f}%)'.format(nonzero, 100.0 * nonzero / rating_matrix.size)
    print 'rating matrix shape: %s' % (rating_matrix.shape,)
    print 'rating matrix:\n%s' % rating_matrix


def reduce_dataset(ratings_df, users_percent):
    user_ids = ratings_df['userId'].unique()

    small_user_size = int(round(len(user_ids) * users_percent, ndigits=0))
    small_user_ids = set(np.random.choice(user_ids, size=small_user_size, replace=False))

    movie_id_set = set()
    row_index_list = []

    for row_index, row in ratings_df.iterrows():
        if row['userId'] in small_user_ids:
            movie_id_set.add(row['movieId'])
            row_index_list.append(row_index)

    print 'num users: %d' % small_user_size
    print 'num movies: {:,d}'.format(len(movie_id_set))
    print 'num ratings: {:,d}'.format(len(row_index_list))
    print 'matrix size: {:,d}'.format(small_user_size * len(movie_id_set))

    small_ratings_df = ratings_df.ix[row_index_list]

    small_ratings_df.to_csv('small_ratings.csv', index=False)

    return ratings_df


def main():
    # ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    # ratings_df = reduce_dataset(ratings_df, users_percent=0.05)

    movie_matrix_data = MovieMatrixData(ratings_df)
    explore_rating_matrix(movie_matrix_data)


if __name__ == '__main__':
    main()
