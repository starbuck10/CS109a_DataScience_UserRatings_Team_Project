import heapq
from collections import defaultdict, namedtuple

import numpy as np
from sklearn.model_selection import train_test_split

from baseline_models import BaselineModel
from common import elapsed_time
from common import get_xy
from read_ratings import read_ratings_df

MovieSimilarity = namedtuple('MovieSimilarity', ['movie_id', 'similarity'])


class MovieSimilarityModel(BaselineModel):
    def __init__(self):
        self.ratings_by_movie = defaultdict(dict)
        self.ratings_by_user = defaultdict(dict)
        self.raters_by_movie = {}
        self.movie_similarity = {}
        self.movie_aij = {}

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

    def calculate_similarity_and_aij(self, movie_id_1, movie_id_2):
        common_raters = self.calculate_common_raters(movie_id_1, movie_id_2)
        support = len(common_raters)
        if support <= 1:
            similarity = 0.0
            aij = 0.0
        else:
            ratings1 = self.get_common_ratings(movie_id_1, common_raters)
            ratings2 = self.get_common_ratings(movie_id_2, common_raters)

            alpha = 4.0

            similarity = support / (np.power(ratings1 - ratings2, 2).sum() + alpha)

            aij = np.multiply(ratings1, ratings2).sum() / support

        return similarity, aij

    def fit(self, ratings_df):
        with elapsed_time('fit'):
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

                    similarity, aij = self.calculate_similarity_and_aij(movie_id_1, movie_id_2)
                    movie_pair = (movie_id_1, movie_id_2)
                    self.movie_similarity[movie_pair] = similarity
                    self.movie_aij[movie_pair] = aij

        return self

    def get_similarity(self, movie_id_1, movie_id_2):
        if movie_id_1 < movie_id_2:
            id_1 = movie_id_1
            id_2 = movie_id_2
        else:
            id_1 = movie_id_2
            id_2 = movie_id_1

        return self.movie_similarity.get((id_1, id_2), -1.0)

    def predict_rating(self, user_id, movie_id):
        ratings = self.ratings_by_user[user_id]

        elements = []

        for movie_id_2 in ratings:
            if movie_id != movie_id_2:
                similarity = self.get_similarity(movie_id, movie_id_2)
                if similarity > 0.0:
                    elements.append(MovieSimilarity(movie_id_2, similarity))

        k = 20

        movie_similarities = heapq.nlargest(k, elements, key=lambda e: e.similarity)

        for movie_similarity in movie_similarities:
            movie_id_2 = movie_similarity.movie_id
            rating = ratings[movie_id_2]

            # print rating, movie_similarity.similarity

        # print ratings[movie_id]

        return 0.0


def build_model(ratings_df):
    model = MovieSimilarityModel()
    train_ratings_df, test_ratings_df = train_test_split(ratings_df)
    model = model.fit(train_ratings_df)

    x_train, y_train = get_xy(train_ratings_df)
    x_test, y_test = get_xy(test_ratings_df)

    y_train_pred = model.predict(x_train)
    # y_test_pred = model.predict(x_test)


def main():
    # ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')

    with elapsed_time('build model'):
        build_model(ratings_df)


if __name__ == '__main__':
    main()
