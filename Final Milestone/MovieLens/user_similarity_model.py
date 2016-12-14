import heapq
from collections import defaultdict
from collections import namedtuple

import numpy as np

from baseline_models import BaselineEffectsModel
from baseline_models import BaselineModel
from common import elapsed_time
from common import score_model
from read_ratings import read_ratings_df_with_timestamp

UserSimilarity = namedtuple('UserSimilarity', ['user_id', 'similarity'])


class UserSimilarityModel(BaselineModel):
    def __init__(self, k_neighbors=40):
        self.k_neighbors = k_neighbors

        self.baseline_model = BaselineEffectsModel()
        self.ratings_by_movie = defaultdict(dict)
        self.ratings_by_user = defaultdict(dict)
        self.movies_by_user = {}
        self.user_similarity = {}

    def set_k_neighbors(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def calculate_common_movies(self, user_id_1, user_id_2):
        movies1 = self.movies_by_user[user_id_1]
        movies2 = self.movies_by_user[user_id_2]
        return movies1 & movies2

    def get_common_ratings(self, user_id, movies):
        all_ratings = self.ratings_by_user[user_id]
        ratings = []
        for movie_id in movies:
            ratings.append(all_ratings[movie_id])

        return np.array(ratings)

    def calculate_similarity(self, user_id_1, user_id_2):
        common_movies = self.calculate_common_movies(user_id_1, user_id_2)
        support = len(common_movies)
        if support <= 1:
            similarity = 0.0
            # aij = 0.0
        else:
            ratings1 = self.get_common_ratings(user_id_1, common_movies)
            ratings2 = self.get_common_ratings(user_id_2, common_movies)

            alpha = 4.0

            similarity = support / (np.power(ratings1 - ratings2, 2).sum() + alpha)

        return similarity

    def fit(self, ratings_df):
        with elapsed_time('fit'):
            self.baseline_model.fit(ratings_df)

            ratings_df = self.baseline_model.create_modified_ratings(ratings_df)

            unique_user_ids = np.array(sorted(ratings_df['userId'].unique()))

            for _, row in ratings_df.iterrows():
                movie_id = row['movieId']
                user_id = row['userId']
                rating = row['rating']
                self.ratings_by_movie[movie_id][user_id] = rating
                self.ratings_by_user[user_id][movie_id] = rating

            for user_id in unique_user_ids:
                self.movies_by_user[user_id] = set(self.ratings_by_user[user_id].keys())

            for user_index_1, user_id_1 in enumerate(unique_user_ids):
                for user_index_2 in xrange(user_index_1 + 1, len(unique_user_ids)):
                    user_id_2 = unique_user_ids[user_index_2]

                    similarity = self.calculate_similarity(user_id_1, user_id_2)
                    user_pair = (user_id_1, user_id_2)
                    self.user_similarity[user_pair] = similarity

        return self

    def get_similarity(self, user_id_1, user_id_2):
        if user_id_1 < user_id_2:
            id_1 = user_id_1
            id_2 = user_id_2
        else:
            id_1 = user_id_2
            id_2 = user_id_1

        return self.user_similarity.get((id_1, id_2), -1.0)

    def clear_predict_caches(self):
        self.zero_prediction_count = 0

    def predict_rating(self, user_id, movie_id):
        ratings = self.ratings_by_movie[movie_id]

        elements = []

        for user_id_2 in ratings:
            if user_id != user_id_2:
                similarity = self.get_similarity(user_id, user_id_2)
                if similarity > 0.0:
                    elements.append(UserSimilarity(user_id_2, similarity))

        user_similarities = heapq.nlargest(self.k_neighbors, elements, key=lambda e: e.similarity)

        if len(user_similarities) > 0:
            similarity_sum = 0.0
            product_sum = 0.0
            for user_similarity in user_similarities:
                user_id_2 = user_similarity.user_id
                rating = ratings[user_id_2]
                similarity = user_similarity.similarity

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


def main():
    ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    # ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    with elapsed_time('build model'):
        score_model(ratings_df, model_f=UserSimilarityModel, model_name='user similarity model')


if __name__ == '__main__':
    main()
