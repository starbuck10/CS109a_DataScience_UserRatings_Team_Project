from collections import defaultdict
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd

EUCLIDEAN = 'euclidean'
MANHATTAN = 'manhattan'
PEARSON = 'pearson'


def read_ratings_df():
    date_parser = lambda time_in_secs: datetime.utcfromtimestamp(float(time_in_secs))
    return pd.read_csv('ml-latest-small/ratings.csv', parse_dates=['timestamp'], date_parser=date_parser)


class MovieData(object):
    def __init__(self):
        self.ratings_df = read_ratings_df()
        self.ratings = defaultdict(dict)
        self.init_ratings()

    def init_ratings(self):
        for _, row in self.ratings_df.iterrows():
            self.ratings[row['userId']][row['movieId']] = row

    def get_movies(self, user_id):
        return set(self.ratings[user_id].keys())

    def get_unique_user_ids(self):
        return self.ratings_df['userId'].unique()

    def get_shared_ratings(self, user1_id, user2_id):
        movies1 = self.get_movies(user1_id)
        movies2 = self.get_movies(user2_id)

        shared_movies = movies1 & movies2

        ratings = {}

        for movie_id in shared_movies:
            ratings[movie_id] = (
                self.ratings[user1_id][movie_id]['rating'],
                self.ratings[user2_id][movie_id]['rating'],
            )

        return ratings

    @staticmethod
    def shared_ratings_to_np_arrays(shared_ratings):
        return np.array(shared_ratings.values()).T

    def get_euclidean_distance(self, user1_id, user2_id):
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)

        if len(shared_ratings) == 0:
            return 0

        ratings = self.shared_ratings_to_np_arrays(shared_ratings)

        ratings1 = ratings[0]
        ratings2 = ratings[1]

        sum_of_squares = np.power(ratings1 - ratings2, 2).sum()

        return 1 / (1 + sqrt(sum_of_squares))

    def get_manhattan_distance(self, user1_id, user2_id):
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)

        if len(shared_ratings) == 0:
            return 0

        ratings = self.shared_ratings_to_np_arrays(shared_ratings)

        ratings1 = ratings[0]
        ratings2 = ratings[1]

        manhattan_sum = np.abs(ratings1 - ratings2).sum()

        return 1 / (1 + manhattan_sum)

    def get_pearson_correlation(self, user1_id, user2_id):
        shared_ratings = self.get_shared_ratings(user1_id, user2_id)

        num_ratings = len(shared_ratings)

        if num_ratings == 0:
            return 0

        ratings = self.shared_ratings_to_np_arrays(shared_ratings)

        ratings1 = ratings[0]
        ratings2 = ratings[1]

        mean1 = ratings1.mean()
        mean2 = ratings2.mean()

        std1 = ratings1.std()
        std2 = ratings2.std()

        if std1 == 0 or std2 == 0:
            return 0

        std_scores_1 = (ratings1 - mean1) / std1
        std_scores_2 = (ratings2 - mean2) / std2

        # numerically stable calculation of the Pearson correlation coefficient

        return abs((std_scores_1 * std_scores_2).sum() / (num_ratings - 1))

    def get_similar_users(self, user_id, metric=EUCLIDEAN):
        metrics = {
            EUCLIDEAN: self.get_euclidean_distance,
            MANHATTAN: self.get_manhattan_distance,
            PEARSON: self.get_pearson_correlation,
        }

        distance_f = metrics[metric]

        similar_users = {}

        for similar_user_id in self.ratings:
            if similar_user_id == user_id:
                continue
            distance = distance_f(user_id, similar_user_id)
            if distance > 0:
                similar_users[similar_user_id] = distance

        return similar_users


def explore_shared_ratings(movie_data):
    unique_user_ids = movie_data.get_unique_user_ids()

    n_pairs = 30
    samples = np.random.choice(unique_user_ids, size=(n_pairs, 2))

    for index, sample in enumerate(samples):
        user1_id = sample[0]
        user2_id = sample[1]

        num_movies_1 = len(movie_data.get_movies(user1_id))
        num_movies_2 = len(movie_data.get_movies(user2_id))

        num_shared_ratings = len(movie_data.get_shared_ratings(user1_id, user2_id))

        print 'pair %2d, user1 movies: %4d, user2 movies: %4d, shared movies: %3d' % (
            index + 1, num_movies_1, num_movies_2, num_shared_ratings)


def explore_distances(movie_data):
    unique_user_ids = movie_data.get_unique_user_ids()

    n_pairs = 30
    samples = np.random.choice(unique_user_ids, size=(n_pairs, 2))

    for index, sample in enumerate(samples):
        user1_id = sample[0]
        user2_id = sample[1]

        num_shared_ratings = len(movie_data.get_shared_ratings(user1_id, user2_id))

        euclidean_distance = movie_data.get_euclidean_distance(user1_id, user2_id)
        manhattan_distance = movie_data.get_manhattan_distance(user1_id, user2_id)
        pearson_correlation = movie_data.get_pearson_correlation(user1_id, user2_id)

        print 'pair %2d, shared movies: %3d, euclidean: %.3f, manhattan: %.3f, pearson: %.3f' % (
            index + 1, num_shared_ratings, euclidean_distance, manhattan_distance, pearson_correlation)


def explore_similar_users(movie_data):
    unique_user_ids = movie_data.get_unique_user_ids()

    n_users = 30
    user_ids = np.random.choice(unique_user_ids, size=n_users, replace=False)

    for index, user_id in enumerate(user_ids):
        similar_users = movie_data.get_similar_users(user_id)

        distances = similar_users.values()

        print 'user %3d, similar users: %d, max similarity: %.3f, mean similarity: %.3f, std similarity: %.3f' % (
            index + 1, len(similar_users), np.max(distances), np.mean(distances), np.std(distances))


def main():
    movie_data = MovieData()

    # explore_shared_ratings(movie_data)
    # explore_distances(movie_data)
    explore_similar_users(movie_data)


if __name__ == '__main__':
    main()
