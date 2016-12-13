import heapq
from collections import defaultdict
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from baseline_models import BaselineEffectsModel
from baseline_models import BaselineModel
from baseline_models import root_mean_squared_error
from common import elapsed_time
from common import get_xy
from read_ratings import read_ratings_df_with_timestamp

MovieSimilarity = namedtuple('MovieSimilarity', ['movie_id', 'similarity'])


class MovieSimilarityModel(BaselineModel):
    def __init__(self, k_neighbors=40):
        self.k_neighbors = k_neighbors

        self.baseline_model = BaselineEffectsModel()
        self.ratings_by_movie = defaultdict(dict)
        self.ratings_by_user = defaultdict(dict)
        self.raters_by_movie = {}
        self.movie_similarity = {}
        # self.movie_aij = {}

    def set_k_neighbors(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def calculate_common_raters(self, movie_id_1, movie_id_2):
        raters1 = self.raters_by_movie[movie_id_1]
        raters2 = self.raters_by_movie[movie_id_2]
        return raters1 & raters2

    def get_common_ratings(self, movie_id, raters):
        all_ratings = self.ratings_by_movie[movie_id]
        ratings = []
        for rater_id in raters:
            ratings.append(all_ratings[rater_id])

        return np.array(ratings)

    def calculate_similarity(self, movie_id_1, movie_id_2):
        common_raters = self.calculate_common_raters(movie_id_1, movie_id_2)
        support = len(common_raters)
        if support <= 1:
            similarity = 0.0
            # aij = 0.0
        else:
            ratings1 = self.get_common_ratings(movie_id_1, common_raters)
            ratings2 = self.get_common_ratings(movie_id_2, common_raters)

            alpha = 4.0

            similarity = support / (np.power(ratings1 - ratings2, 2).sum() + alpha)

            # aij = np.multiply(ratings1, ratings2).sum() / support

        return similarity

    def fit(self, ratings_df):
        with elapsed_time('fit'):
            self.baseline_model.fit(ratings_df)

            ratings_df = self.baseline_model.create_modified_ratings(ratings_df)

            unique_movie_ids = np.array(sorted(ratings_df['movieId'].unique()))

            for _, row in ratings_df.iterrows():
                movie_id = row['movieId']
                user_id = row['userId']
                rating = row['rating']
                self.ratings_by_movie[movie_id][user_id] = rating
                self.ratings_by_user[user_id][movie_id] = rating

            for movie_id in unique_movie_ids:
                self.raters_by_movie[movie_id] = set(self.ratings_by_movie[movie_id].keys())

            for movie_index_1, movie_id_1 in enumerate(unique_movie_ids):
                for movie_index_2 in xrange(movie_index_1 + 1, len(unique_movie_ids)):
                    movie_id_2 = unique_movie_ids[movie_index_2]

                    similarity = self.calculate_similarity(movie_id_1, movie_id_2)
                    movie_pair = (movie_id_1, movie_id_2)
                    self.movie_similarity[movie_pair] = similarity
                    # self.movie_aij[movie_pair] = aij

        return self

    def get_similarity(self, movie_id_1, movie_id_2):
        if movie_id_1 < movie_id_2:
            id_1 = movie_id_1
            id_2 = movie_id_2
        else:
            id_1 = movie_id_2
            id_2 = movie_id_1

        return self.movie_similarity.get((id_1, id_2), -1.0)

    def clear_predict_caches(self):
        self.zero_prediction_count = 0

    def predict_rating(self, user_id, movie_id):
        ratings = self.ratings_by_user[user_id]

        elements = []

        for movie_id_2 in ratings:
            if movie_id != movie_id_2:
                similarity = self.get_similarity(movie_id, movie_id_2)
                if similarity > 0.0:
                    elements.append(MovieSimilarity(movie_id_2, similarity))

        movie_similarities = heapq.nlargest(self.k_neighbors, elements, key=lambda e: e.similarity)

        if len(movie_similarities) > 0:
            similarity_sum = 0.0
            product_sum = 0.0
            for movie_similarity in movie_similarities:
                movie_id_2 = movie_similarity.movie_id
                rating = ratings[movie_id_2]
                similarity = movie_similarity.similarity

                product_sum += similarity * rating
                similarity_sum += similarity

            rating = product_sum / similarity_sum
        else:
            rating = 0.0
            self.zero_prediction_count += 1

        result = self.baseline_model.predict_baseline_rating(user_id, movie_id) + rating

        return result

    def predict(self, x):
        self.clear_predict_caches()
        predictions = [self.predict_rating(row['userId'], row['movieId']) for _, row in x.iterrows()]
        print 'used baseline predictions: %.1f%%' % (100.0 * self.zero_prediction_count / len(predictions))
        return predictions


def show_scores_plot(k_neighbors_values, val_scores, train_scores):
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.plot(k_neighbors_values, val_scores, label='validation')
    ax.plot(k_neighbors_values, train_scores, label='train')

    ax.set_xlabel('k_neighbors')
    ax.set_ylabel('$R^2$')
    ax.set_title('Test and validation scores for different k_neighbors values (movie similarity model)')

    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def build_model(ratings_df):
    train_val_ratings_df, test_ratings_df = train_test_split(ratings_df)

    train_ratings_df, validation_ratings_df = train_test_split(train_val_ratings_df)

    best_score = -float('inf')
    best_k_neighbors = None

    model = MovieSimilarityModel()

    model = model.fit(train_ratings_df)

    k_neighbors_values = [1, 5, 10, 20, 30, 40, 50, 75, 100]

    val_scores = []
    train_scores = []

    for k_neighbors in k_neighbors_values:
        model.set_k_neighbors(k_neighbors=k_neighbors)

        x_train, y_train = get_xy(train_ratings_df)
        x_val, y_val = get_xy(validation_ratings_df)

        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)

        train_score = r2_score(y_train, y_train_pred)
        val_score = r2_score(y_val, y_val_pred)

        if val_score > best_score:
            best_score = val_score
            best_k_neighbors = k_neighbors

        val_scores.append(val_score)
        train_scores.append(train_score)

        print 'k: %d, validation score: %.5f, train score: %.5f\n' % (k_neighbors, val_score, train_score)

    print 'best k: %d, best score: %.5f' % (best_k_neighbors, best_score)

    model = MovieSimilarityModel(k_neighbors=best_k_neighbors)

    model = model.fit(train_val_ratings_df)

    x_train_val, y_train_val = get_xy(train_val_ratings_df)
    x_test, y_test = get_xy(test_ratings_df)

    y_train_val_pred = model.predict(x_train_val)
    y_test_pred = model.predict(x_test)

    train_val_score = r2_score(y_train_val, y_train_val_pred)
    test_score = r2_score(y_test, y_test_pred)

    train_val_rmse = root_mean_squared_error(y_train_val, y_train_val_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    print 'train score: %.4f, test score: %.4f' % (train_val_score, test_score)
    print 'train rmse: %.4f, test rmse: %.4f' % (train_val_rmse, test_rmse)

    show_scores_plot(k_neighbors_values, val_scores, train_scores)


def main():
    ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    # ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    with elapsed_time('build model'):
        build_model(ratings_df)


if __name__ == '__main__':
    main()
