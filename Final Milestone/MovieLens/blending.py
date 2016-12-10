from collections import Counter

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from baseline_models import BaselineEffectsModel
from baseline_models import BaselineMeansModel
from baseline_models import root_mean_squared_error
from common import elapsed_time
from common import get_xy
from read_ratings import read_ratings_df


class BlendingModel(object):
    def __init__(self):
        self.models = [
            BaselineMeansModel(user_weight=0.0),
            BaselineMeansModel(user_weight=1.0),
            BaselineEffectsModel(movie_lambda=5.0, user_lambda=20.0),
        ]
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

            print Counter(supports)

            self.regression.fit(blend_predictions, y)

            print self.regression.coef_, self.regression.intercept_

    def predict(self, x):
        blend_predictions = self.get_blend_predictions(x)

        return self.regression.predict(blend_predictions)


def build_blending_model(ratings_df):
    model = BlendingModel()

    train_ratings_df, test_ratings_df = train_test_split(ratings_df)
    x_train, y_train = get_xy(train_ratings_df)
    x_test, y_test = get_xy(test_ratings_df)

    model.fit(train_ratings_df)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_r2_score = r2_score(y_train, y_train_pred)
    test_r2_score = r2_score(y_test, y_test_pred)

    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    print 'train r2 score: %.4f, test r2 score: %.4f' % (train_r2_score, test_r2_score)
    print 'train rmse: %.4f, test rmse: %.4f' % (train_rmse, test_rmse)


def main():
    # ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    build_blending_model(ratings_df)


if __name__ == '__main__':
    main()
