import math

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from common import elapsed_time
from common import get_xy


def root_mean_squared_error(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))


class BaselineModel(object):
    def predict_rating(self, user_id, movie_id):
        pass

    def predict(self, x):
        return [self.predict_rating(row['userId'], row['movieId']) for _, row in x.iterrows()]

    def score(self, x, y):
        return r2_score(y, self.predict(x))


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
        self.user_groups = None

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
        with elapsed_time('effects init'):
            _, y_train = get_xy(ratings_df)
            self.y_mean = y_train.mean()

            movie_ratings = ratings_df.groupby('movieId')['rating']
            self.user_groups = ratings_df.groupby('userId')

            self.movie_effects = self.calculate_movie_effects(movie_ratings)
            self.user_effects = self.calculate_user_effects(self.user_groups)

        return self

    def create_modified_ratings(self, ratings_df):
        ratings_df = ratings_df.copy()

        for index, row in ratings_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']
            pred_rating = self.predict_baseline_rating(user_id, movie_id)

            residual = rating - pred_rating

            ratings_df.loc[index, 'rating'] = residual

        return ratings_df

    def predict_baseline_rating(self, user_id, movie_id):
        return self.y_mean + self.movie_effects.get(movie_id, 0.0) + self.user_effects[user_id]

    def predict_rating(self, user_id, movie_id):
        return self.predict_baseline_rating(user_id, movie_id)
