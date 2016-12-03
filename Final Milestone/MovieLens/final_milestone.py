from collections import defaultdict
from datetime import datetime
from math import sqrt

import numpy as np
import pandas as pd


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

    def get_shared_ratings(self, user1_id, user2_id):
        movies1 = set(self.ratings[user1_id].keys())
        movies2 = set(self.ratings[user2_id].keys())

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

        return abs((std_scores_1 * std_scores_2).sum() / (num_ratings - 1))


def explore_shared_ratings(movie_data):
    unique_user_ids = movie_data.ratings_df['userId'].unique()

    n_pairs = 15
    samples = np.random.choice(unique_user_ids, size=(n_pairs, 2))

    for sample in samples:
        user1_id = sample[0]
        user2_id = sample[1]
        euclidean_distance = movie_data.get_euclidean_distance(user1_id, user2_id)
        manhattan_distance = movie_data.get_manhattan_distance(user1_id, user2_id)
        pearson_correlation = movie_data.get_pearson_correlation(user1_id, user2_id)
        print '%.3f %.3f %.3f' % (euclidean_distance, manhattan_distance, pearson_correlation)


def main():
    movie_data = MovieData()

    explore_shared_ratings(movie_data)


if __name__ == '__main__':
    main()
