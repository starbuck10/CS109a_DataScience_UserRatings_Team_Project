from collections import defaultdict
from math import sqrt

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from common import get_xy
from read_ratings import read_ratings_df_with_timestamp

EUCLIDEAN = 'euclidean'
MANHATTAN = 'manhattan'
PEARSON = 'pearson'


def get_scores(y_test, y_test_pred, y_train, y_train_pred):
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return train_score, test_score


class MovieData(object):
    def __init__(self):
        self.ratings_df = read_ratings_df_with_timestamp()
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

    def predict_score(self, user_id, movie_id):
        similar_users = self.get_similar_users(user_id)

        total_rating_sum = 0
        similarity_sum = 0

        for similar_user_id, similarity in similar_users.items():
            user_ratings = self.ratings[similar_user_id]
            if movie_id in user_ratings:
                total_rating_sum += similarity * user_ratings[movie_id]['rating']
                similarity_sum += similarity

        if similarity_sum == 0:
            return 0

        return total_rating_sum / similarity_sum


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
        similar_users = movie_data.similar_users[user_id]

        distances = similar_users.values()

        print 'user %3d, similar users: %d, max similarity: %.3f, mean: %.3f, std: %.3f' % (
            index + 1, len(similar_users), np.max(distances), np.mean(distances), np.std(distances))


def explore_predict_score(movie_data):
    ratings_df = movie_data.ratings_df
    rating_indices = ratings_df.index

    n_ratings = 30
    sample = np.random.choice(rating_indices, size=n_ratings, replace=False)

    for index, rating_index in enumerate(sample):
        row = ratings_df.ix[rating_index]

        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']

        score = movie_data.predict_score(user_id, movie_id)

        print 'rating %2d, rating: %.1f, predicted: %.3f' % (index + 1, rating, score)


def get_y_pred_user_similarity_model(movie_data, x):
    return [movie_data.predict_score(row['userId'], row['movieId']) for _, row in x.iterrows()]


def get_score_user_similarity_model(movie_data):
    ratings_df = movie_data.ratings_df

    train_scores = []
    test_scores = []
    n_iter = 1
    for _ in xrange(n_iter):
        train_df, test_df = train_test_split(ratings_df)

        x_train, y_train = get_xy(train_df)
        x_test, y_test = get_xy(test_df)

        y_train_pred = get_y_pred_user_similarity_model(movie_data, x_train)
        y_test_pred = get_y_pred_user_similarity_model(movie_data, x_test)

        train_score, test_score = get_scores(y_test, y_test_pred, y_train, y_train_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)

    print 'mean train score: %.4f, std: %.4f' % (np.mean(train_scores), np.std(train_scores))
    print 'mean test score: %.4f, std: %.4f' % (np.mean(test_scores), np.std(test_scores))


def main():
    movie_data = MovieData()

    # explore_shared_ratings(movie_data)
    # explore_distances(movie_data)
    # explore_similar_users(movie_data)

    explore_predict_score(movie_data)
    # get_score_user_similarity_model(movie_data)


if __name__ == '__main__':
    main()
