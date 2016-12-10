import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from common import get_xy
from read_ratings import read_ratings_df


def get_scores(y_test, y_test_pred, y_train, y_train_pred):
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return train_score, test_score


class MovieMatrixData(object):
    def __init__(self, ratings_df):
        start = time.time()

        self.ratings_df = ratings_df
        self.users = np.array(sorted(ratings_df['userId'].unique()))
        self.movies = np.array(sorted(ratings_df['movieId'].unique()))

        self.mean_user_ratings = ratings_df.groupby('userId')['rating'].mean()
        self.mean_movie_ratings = ratings_df.groupby('movieId')['rating'].mean()

        ratings = self.ratings_df['rating']
        self.min_rating = ratings.min()
        self.max_rating = ratings.max()

        self.user_id_to_index_dict = {user_id: index for index, user_id in enumerate(self.users)}
        self.movie_id_to_index_dict = {movie_id: index for index, movie_id in enumerate(self.movies)}

        self.rating_matrix = np.zeros((len(self.users), len(self.movies)))

        for _, row in self.ratings_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']

            user_index = self.get_user_index(user_id)
            movie_index = self.get_movie_index(movie_id)

            self.rating_matrix[user_index, movie_index] = self.to_normalized_rating(user_id, movie_id, row['rating'])

        elapsed = time.time() - start
        print 'matrix init: %.2f secs' % elapsed

    def to_normalized_rating(self, user_id, movie_id, rating):
        mean_user_rating = self.mean_user_ratings[user_id]
        mean_movie_rating = self.mean_movie_ratings[movie_id]
        return rating - 0.5 * (mean_user_rating + mean_movie_rating)

    def to_rating(self, user_id, movie_id, normalized_rating):
        mean_user_rating = self.mean_user_ratings[user_id]
        mean_movie_rating = self.mean_movie_ratings[movie_id]
        rating = normalized_rating + 0.5 * (mean_user_rating + mean_movie_rating)
        rating = min(rating, self.max_rating)
        rating = max(rating, self.min_rating)
        return rating

    def get_user_index(self, user_id):
        return self.user_id_to_index_dict[user_id]

    def get_movie_index(self, movie_id):
        return self.movie_id_to_index_dict.get(movie_id, -1)


def explore_rating_matrix(movie_matrix_data):
    rating_matrix = movie_matrix_data.rating_matrix

    nonzero = np.count_nonzero(rating_matrix)

    print 'rating matrix size: {:,d}'.format(rating_matrix.size)
    print 'nonzero elements: {:,d} (density: {:.2f}%)'.format(nonzero, 100.0 * nonzero / rating_matrix.size)
    print 'rating matrix shape: %s' % (rating_matrix.shape,)
    print 'rating matrix:\n%s' % rating_matrix


def reduce_dataset(ratings_df, users_percent):
    user_ids = ratings_df['userId'].unique()

    small_user_size = int(round(len(user_ids) * users_percent, ndigits=0))
    small_user_ids = set(np.random.choice(user_ids, size=small_user_size, replace=False))

    movie_id_set = set()
    row_index_list = []

    for row_index, row in ratings_df.iterrows():
        if row['userId'] in small_user_ids:
            movie_id_set.add(row['movieId'])
            row_index_list.append(row_index)

    print 'num users: %d' % small_user_size
    print 'num movies: {:,d}'.format(len(movie_id_set))
    print 'num ratings: {:,d}'.format(len(row_index_list))
    print 'matrix size: {:,d}'.format(small_user_size * len(movie_id_set))

    small_ratings_df = ratings_df.ix[row_index_list]

    small_ratings_df.to_csv('small_ratings.csv', index=False)

    return ratings_df


def init_pq_matrices(rating_matrix, k_features):
    n_rows, m_cols = rating_matrix.shape

    c = 0.1
    p_matrix = np.random.uniform(low=-c, high=c, size=(n_rows, k_features))
    q_matrix = np.random.uniform(low=-c, high=c, size=(k_features, m_cols))

    return p_matrix, q_matrix


def get_eij(rating_matrix, p_matrix, q_matrix, i, j):
    return rating_matrix[i, j] - p_matrix[i].dot(q_matrix[:, j])


def print_progress(start_time, step, e):
    elapsed_time = time.time() - start_time
    print '%4d: %8.2f %6.1f' % (step, e, elapsed_time)


def factor(rating_matrix, k_features=2):
    start_time = time.time()

    num_steps = 5000
    learning_rate = 0.0002
    regularization_param = 0.04

    p_matrix, q_matrix = init_pq_matrices(rating_matrix, k_features)

    n_rows, m_cols = rating_matrix.shape
    for step in xrange(num_steps):
        for i in xrange(n_rows):
            for j in xrange(m_cols):
                if rating_matrix[i, j] > 0:
                    eij = get_eij(rating_matrix, p_matrix, q_matrix, i, j)
                    for k in xrange(k_features):
                        p_matrix[i, k] += learning_rate * (
                            2 * eij * q_matrix[k, j] - regularization_param * p_matrix[i, k])
                        q_matrix[k, j] += learning_rate * (
                            2 * eij * p_matrix[i, k] - regularization_param * q_matrix[k, j])

        e = 0.0
        for i in xrange(n_rows):
            for j in xrange(m_cols):
                if rating_matrix[i, j] > 0:
                    eij = get_eij(rating_matrix, p_matrix, q_matrix, i, j)
                    e += pow(eij, 2)
                    for k in xrange(k_features):
                        e += regularization_param / 2.0 * (pow(p_matrix[i, k], 2) + pow(q_matrix[k, j], 2))

        if step % 200 == 0:
            print_progress(start_time, step, e)

        if e < 0.001:
            break

    print_progress(start_time, step, e)

    return p_matrix, q_matrix


def predict(movie_matrix_data, model, user_id, movie_id):
    movie_index = movie_matrix_data.get_movie_index(movie_id)
    if movie_index < 0:
        pred_y = movie_matrix_data.mean_user_ratings[user_id]
    else:
        user_index = movie_matrix_data.get_user_index(user_id)
        normalized_rating = model[user_index, movie_index]

        pred_y = movie_matrix_data.to_rating(user_id, movie_id, normalized_rating)

    return pred_y


def get_y_pred(movie_matrix_data, model, x):
    return [predict(movie_matrix_data, model, row['userId'], row['movieId']) for _, row in x.iterrows()]


def build_model(ratings_df):
    train_ratings_df, test_ratings_df = train_test_split(ratings_df)

    x_train, y_train = get_xy(train_ratings_df)
    x_test, y_test = get_xy(test_ratings_df)

    movie_matrix_data = MovieMatrixData(train_ratings_df)
    explore_rating_matrix(movie_matrix_data)

    rating_matrix = movie_matrix_data.rating_matrix

    p_matrix, q_matrix = factor(rating_matrix)

    model = p_matrix.dot(q_matrix)

    print model.shape

    print np.amin(model), np.amax(model)

    model = np.clip(model, a_min=movie_matrix_data.min_rating, a_max=movie_matrix_data.max_rating)

    y_train_pred = get_y_pred(movie_matrix_data, model, x_train)
    y_test_pred = get_y_pred(movie_matrix_data, model, x_test)

    train_score, test_score = get_scores(y_test, y_test_pred, y_train, y_train_pred)

    print 'train: %.3f, test: %.3f' % (train_score, test_score)


def main():
    # ratings_df = read_ratings_df_with_timestamp('ml-latest-small/ratings.csv')
    # ratings_df = read_ratings_df('ml-latest-small/ratings_5_pct.csv')
    ratings_df = read_ratings_df('ml-latest-small/ratings_10_pct.csv')

    # ratings_df = reduce_dataset(ratings_df, users_percent=0.1)

    build_model(ratings_df)


if __name__ == '__main__':
    main()
