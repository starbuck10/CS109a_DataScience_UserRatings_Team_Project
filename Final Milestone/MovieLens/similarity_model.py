import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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


def get_xy(ratings_df):
    y = ratings_df['rating']
    x = ratings_df.drop('rating', axis=1)
    return x, y


class BaselineModel(object):
    def predict_rating(self, user_id, movie_id):
        pass

    def predict(self, x):
        return [self.predict_rating(row['userId'], row['movieId']) for _, row in x.iterrows()]

    def score(self, x, y):
        return r2_score(y, self.predict(x))

    def root_mean_squared_error(self, x, y):
        return math.sqrt(mean_squared_error(y, self.predict(x)))


class BaselineTotalMeanModel(BaselineModel):
    def __init__(self):
        self.y_mean = None

    def fit(self, ratings_df):
        _, y_train = get_xy(ratings_df)
        self.y_mean = y_train.mean()
        return self

    def predict_rating(self, user_id, movie_id):
        return self.y_mean


class BaselineMeansModel(BaselineModel):
    def __init__(self, user_weight=0.5):
        self.user_weight = user_weight
        self.mean_user_ratings = None
        self.mean_movie_ratings = None

    def fit(self, ratings_df):
        self.mean_user_ratings = ratings_df.groupby('userId')['rating'].mean()
        self.mean_movie_ratings = ratings_df.groupby('movieId')['rating'].mean()
        return self

    def predict_rating(self, user_id, movie_id):
        user_rating = self.mean_user_ratings[user_id]
        movie_rating = self.mean_movie_ratings.get(movie_id, user_rating)
        return self.user_weight * user_rating + (1.0 - self.user_weight) * movie_rating


class BaselineEffectsModel(BaselineModel):
    def __init__(self, movie_lambda=5.0, user_lambda=20.0):
        self.movie_lambda = movie_lambda
        self.user_lambda = user_lambda

        self.y_mean = None
        self.movie_effects = None
        self.user_effects = None

    def calculate_movie_effect(self, ratings):
        return (ratings - self.y_mean).sum() / (self.movie_lambda + len(ratings))

    def calculate_movie_effects(self, movie_ratings):
        return movie_ratings.agg(lambda ratings: self.calculate_movie_effect(ratings))

    def calculate_user_effect(self, ratings_df):
        s = 0.0
        for _, row in ratings_df.iterrows():
            s += row['rating'] - self.y_mean - self.movie_effects[row['movieId']]

        return s / (self.user_lambda + len(ratings_df))

    def calculate_user_effects(self, user_groups):
        user_ids = []
        user_effects = []

        for user_id, group in user_groups:
            user_effect = self.calculate_user_effect(group)

            user_ids.append(user_id)
            user_effects.append(user_effect)

        return pd.Series(user_effects, index=user_ids)

    def fit(self, ratings_df):
        _, y_train = get_xy(ratings_df)
        self.y_mean = y_train.mean()

        movie_ratings = ratings_df.groupby('movieId')['rating']
        user_groups = ratings_df.groupby('userId')

        self.movie_effects = self.calculate_movie_effects(movie_ratings)
        self.user_effects = self.calculate_user_effects(user_groups)

        return self

    def predict_rating(self, user_id, movie_id):
        return self.y_mean + self.movie_effects.get(movie_id, 0.0) + self.user_effects[user_id]


def build_model(ratings_df):
    train_scores = []
    test_scores = []
    train_rmse_scores = []
    test_rmse_scores = []
    n_iter = 2

    # model = BaselineTotalMeanModel()
    # model = BaselineMeansModel(user_weight=0.5)
    model = BaselineEffectsModel(movie_lambda=5.0, user_lambda=20.0)

    for _ in xrange(n_iter):
        train_ratings_df, test_ratings_df = train_test_split(ratings_df)

        model = model.fit(train_ratings_df)

        x_train, y_train = get_xy(train_ratings_df)
        x_test, y_test = get_xy(test_ratings_df)

        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)

        train_rmse = model.root_mean_squared_error(x_train, y_train)
        test_rmse = model.root_mean_squared_error(x_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

        train_rmse_scores.append(train_rmse)
        test_rmse_scores.append(test_rmse)

    print 'mean train score: %.4f, std: %.4f' % (np.mean(train_scores), np.std(train_scores))
    print 'mean test score: %.4f, std: %.4f' % (np.mean(test_scores), np.std(test_scores))
    print
    print 'mean train rmse: %.4f, std: %.4f' % (np.mean(train_rmse_scores), np.std(train_rmse_scores))
    print 'mean test rmse: %.4f, std: %.4f' % (np.mean(test_rmse_scores), np.std(test_rmse_scores))


def main():
    # ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    build_model(ratings_df)


if __name__ == '__main__':
    main()
