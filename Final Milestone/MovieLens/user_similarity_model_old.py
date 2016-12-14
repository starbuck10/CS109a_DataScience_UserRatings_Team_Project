import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from baseline_models import BaselineEffectsModel
from baseline_models import root_mean_squared_error
from common import elapsed_time
from common import get_xy
from read_ratings import read_ratings_df


class UserSimilarityModel(BaselineEffectsModel):
    def __init__(self, movie_lambda=5.0, user_lambda=20.0):
        super(UserSimilarityModel, self).__init__(movie_lambda, user_lambda)

        self.user_id_to_index_dict = None
        self.movie_id_to_index_dict = None
        self.rating_matrix = None
        self.users = None

    def get_user_index(self, user_id):
        return self.user_id_to_index_dict[user_id]

    def get_movie_index(self, movie_id):
        return self.movie_id_to_index_dict.get(movie_id, -1)

    def get_distance(self, user_id_1, user_id_2):
        v1 = self.rating_matrix[self.get_user_index(user_id_1)]
        v2 = self.rating_matrix[self.get_user_index(user_id_2)]

        distance = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))

        return distance[0][0]

    def get_similar_users(self, user_id):
        user_ids = []
        for other_user_id in self.users:
            if user_id == other_user_id:
                continue
            distance = self.get_distance(user_id, other_user_id)
            if distance > 0.05:
                user_ids.append(other_user_id)

        return user_ids

    def fit(self, ratings_df):
        super(UserSimilarityModel, self).fit(ratings_df)

        self.users = np.array(sorted(ratings_df['userId'].unique()))
        movies = np.array(sorted(ratings_df['movieId'].unique()))

        mean_user_ratings = self.user_groups['rating'].mean()

        self.user_id_to_index_dict = {user_id: index for index, user_id in enumerate(self.users)}
        self.movie_id_to_index_dict = {movie_id: index for index, movie_id in enumerate(movies)}

        self.rating_matrix = np.zeros((len(self.users), len(movies)))

        for _, row in ratings_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']

            user_index = self.get_user_index(user_id)
            movie_index = self.get_movie_index(movie_id)

            rating = row['rating']
            normalized_rating = rating - mean_user_ratings[user_id]
            self.rating_matrix[user_index, movie_index] = normalized_rating

        # random_users = np.random.choice(self.users, size=1)
        #
        # for user_id in random_users:
        #     movie_id = np.random.choice(movies, size=1)[0]
        #     rating = self.predict_rating_2(user_id, movie_id)

        return self

    def predict_rating(self, user_id, movie_id):
        return self.y_mean + self.movie_effects.get(movie_id, 0.0) + self.user_effects[user_id]

    def predict_rating_2(self, user_id, movie_id):
        movie_index = self.get_movie_index(movie_id)

        similar_users = self.get_similar_users(user_id)

        similarity_ratings = []
        for similar_user_id in similar_users:
            user_index = self.get_user_index(similar_user_id)
            rating = self.rating_matrix[user_index, movie_index]
            if rating != 0.0:
                similarity_ratings.append(rating)

        similarity_rating = 0.0 * (np.mean(similarity_ratings) if len(similarity_ratings) > 0 else 0.0)

        return self.y_mean + self.movie_effects.get(movie_id, 0.0) + self.user_effects[user_id] + similarity_rating


def build_model(ratings_df):
    train_scores = []
    test_scores = []
    train_rmse_scores = []
    test_rmse_scores = []
    n_iter = 1

    # model = BaselineTotalMeanModel()
    # model = BaselineMeansModel(user_weight=0.5)
    # model = BaselineEffectsModel(movie_lambda=5.0, user_lambda=20.0)
    model = UserSimilarityModel(movie_lambda=5.0, user_lambda=20.0)

    for _ in xrange(n_iter):
        train_ratings_df, test_ratings_df = train_test_split(ratings_df)

        model = model.fit(train_ratings_df)

        x_train, y_train = get_xy(train_ratings_df)
        x_test, y_test = get_xy(test_ratings_df)

        with elapsed_time('scoring'):
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            train_rmse = root_mean_squared_error(y_train, y_train_pred)
            test_rmse = root_mean_squared_error(y_test, y_test_pred)

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
