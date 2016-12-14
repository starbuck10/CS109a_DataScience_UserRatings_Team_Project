from collections import namedtuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from baseline_models import BaselineEffectsModel
from baseline_models import BaselineMeansModel
from common import elapsed_time
from common import get_xy
from common import root_mean_squared_error
from movie_similarity_model import MovieSimilarityModel
from read_ratings import read_ratings_df_with_timestamp
from user_similarity_model import UserSimilarityModel


class BlendingModel(object):
    def __init__(self, models):
        self.models = models
        self.regression = LinearRegression()

    def get_blend_predictions(self, x):
        predictions = []
        for model in self.models:
            y_pred = model.predict(x)
            predictions.append(y_pred)
        pred_arr = np.array(predictions).T
        return pred_arr

    def fit(self, ratings_df):
        with elapsed_time('total fit'):
            for model in self.models:
                model.fit(ratings_df)

            x, y = get_xy(ratings_df)

            with elapsed_time('get blend predictions'):
                blend_predictions = self.get_blend_predictions(x)

            user_groups = x.groupby('userId')
            movie_groups = x.groupby('movieId')
            supports = []
            for _, row in x.iterrows():
                user_id = row['userId']
                movie_id = row['movieId']
                user_support = len(user_groups.get_group(user_id))
                movie_support = len(movie_groups.get_group(movie_id))
                support = min(user_support, movie_support)
                supports.append(support)

            # print Counter(supports)

            self.regression.fit(blend_predictions, y)

            print 'linear regression coefficients: %s, intercept: %.3f' % (
                self.regression.coef_, self.regression.intercept_)

    def predict(self, x):
        blend_predictions = self.get_blend_predictions(x)

        return self.regression.predict(blend_predictions)


ModelRecord = namedtuple('ModelRecord', ['name', 'model'])


def score_models(ratings_df, model_records):
    train_ratings_df, test_ratings_df = train_test_split(ratings_df)
    x_train, y_train = get_xy(train_ratings_df)
    x_test, y_test = get_xy(test_ratings_df)

    print

    for model_record in model_records:
        model_name, model = model_record
        model.fit(train_ratings_df)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        train_r2_score = r2_score(y_train, y_train_pred)
        test_r2_score = r2_score(y_test, y_test_pred)

        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)

        print '%s' % model_name
        print 'train r2 score: %.4f, test r2 score: %.4f' % (train_r2_score, test_r2_score)
        print 'train rmse: %.4f, test rmse: %.4f\n' % (train_rmse, test_rmse)


def score_baseline_mean_blending_model(ratings_df):
    blend_models = [
        BaselineMeansModel(user_weight=0.0),
        BaselineMeansModel(user_weight=1.0),
    ]
    models = [
        ModelRecord('movie mean model', BaselineMeansModel(user_weight=0.0)),
        ModelRecord('user mean model', BaselineMeansModel(user_weight=1.0)),
        ModelRecord('blending model (user & movie mean models)', BlendingModel(blend_models)),
    ]
    score_models(ratings_df, models)


def score_include_all_blending_model(ratings_df):
    blend_models = [
        BaselineMeansModel(user_weight=0.0),
        BaselineMeansModel(user_weight=1.0),
        BaselineEffectsModel(movie_lambda=5.0, user_lambda=20.0),
        MovieSimilarityModel(k_neighbors=40),
        UserSimilarityModel(k_neighbors=30),
    ]
    models = [
        ModelRecord('blending model (user & movie mean models)', BlendingModel(blend_models)),
    ]
    score_models(ratings_df, models)


def main():
    ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    # ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    # score_baseline_mean_blending_model(ratings_df)
    score_include_all_blending_model(ratings_df)


if __name__ == '__main__':
    main()
