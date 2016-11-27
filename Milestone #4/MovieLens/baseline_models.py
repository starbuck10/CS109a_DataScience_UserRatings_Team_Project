from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score


def date_parse(time_in_secs):
    return datetime.utcfromtimestamp(float(time_in_secs))


def read_data():
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv', parse_dates=['timestamp'], date_parser=date_parse)
    return ratings_df


def get_fig_size(nrows=1):
    return 15, 10 * nrows


def show_ratings_histogram(ratings):
    _, ax = plt.subplots(1, 1, figsize=get_fig_size())

    ax.hist(ratings, bins=np.arange(0.25, 5.5, step=0.5), alpha=0.4)

    ratings_mean = ratings.mean()

    ax.axvline(x=ratings_mean, linewidth=3, color='k')
    plt.text(ratings_mean - 0.55, 25000, 'mean = %.2f' % ratings_mean)

    ax.set_xlabel('rating')
    ax.set_ylabel('count')
    ax.set_title('Overall ratings')

    plt.tight_layout()
    plt.show()


def explore_total_mean_rating(ratings_df):
    print 'ratings_df.shape: %s\n' % (ratings_df.shape,)

    print 'raw ratings data:'
    display(ratings_df.head())
    print

    ratings = ratings_df['rating']

    print 'mean rating: %.2f, std: %.2f' % (ratings.mean(), ratings.std())

    show_ratings_histogram(ratings)


def get_xy(ratings_df):
    y = ratings_df['rating']
    x = ratings_df.drop('rating', axis=1)
    return x, y


def get_y_pred_total_mean_model(y_mean, x):
    shape = (len(x),)
    return np.full(shape, y_mean)


def get_scores(y_test, y_test_pred, y_train, y_train_pred):
    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    return train_score, test_score


def fit_total_mean_model(ratings_df):
    train_scores = []
    test_scores = []
    n_iter = 100
    for _ in xrange(n_iter):
        train_df, test_df = train_test_split(ratings_df)

        x_train, y_train = get_xy(train_df)
        x_test, y_test = get_xy(test_df)

        y_mean = y_train.mean()

        y_train_pred = get_y_pred_total_mean_model(y_mean, x_train)
        y_test_pred = get_y_pred_total_mean_model(y_mean, x_test)

        train_score, test_score = get_scores(y_test, y_test_pred, y_train, y_train_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)

    print 'mean train score: %f, std: %f' % (np.mean(train_scores), np.std(train_scores))
    print 'mean test score: %f, std: %f' % (np.mean(test_scores), np.std(test_scores))


def get_mean_user_ratings(train_df):
    return train_df.groupby('userId')['rating'].mean()


def show_user_mean_ratings_histogram(ratings_df):
    user_ratings = get_mean_user_ratings(ratings_df)

    print 'The maximum user mean rating: %.2f' % user_ratings.max()
    user_ratings_mean = user_ratings.mean()
    print 'The mean user mean rating: %.2f' % user_ratings_mean
    print 'The minimum user mean rating: %.2f' % user_ratings.min()

    _, ax = plt.subplots(1, 1, figsize=get_fig_size())

    ax.hist(user_ratings, bins=50, alpha=0.4)

    ax.axvline(x=user_ratings_mean, linewidth=3, color='k')
    plt.text(user_ratings_mean + 0.05, 57, 'mean = %.2f' % user_ratings_mean)

    ax.set_xlabel('user mean rating')
    ax.set_ylabel('count')
    ax.set_title('User mean ratings')

    plt.tight_layout()
    plt.show()


def explore_mean_user_ratings(ratings_df):
    user_ids = ratings_df['userId']

    print 'How many users?', len(set(user_ids))

    show_user_mean_ratings_histogram(ratings_df)


def get_y_pred_mean_user_ratings_model(mean_user_ratings, x):
    return [mean_user_ratings[row['userId']] for _, row in x.iterrows()]


def fit_mean_user_ratings_model(ratings_df):
    train_scores = []
    test_scores = []
    n_iter = 4
    for _ in xrange(n_iter):
        train_df, test_df = train_test_split(ratings_df)

        x_train, y_train = get_xy(train_df)
        x_test, y_test = get_xy(test_df)

        mean_user_ratings = get_mean_user_ratings(train_df)

        y_train_pred = get_y_pred_mean_user_ratings_model(mean_user_ratings, x_train)
        y_test_pred = get_y_pred_mean_user_ratings_model(mean_user_ratings, x_test)

        train_score, test_score = get_scores(y_test, y_test_pred, y_train, y_train_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)

    print 'mean train score: %.4f, std: %.4f' % (np.mean(train_scores), np.std(train_scores))
    print 'mean test score: %.4f, std: %.4f' % (np.mean(test_scores), np.std(test_scores))


def get_mean_movie_ratings(ratings_df):
    return ratings_df.groupby('movieId')['rating'].mean()


def show_movie_mean_ratings_histogram(ratings_df):
    movie_mean_ratings = get_mean_movie_ratings(ratings_df)

    print 'The maximum movie mean rating: %.2f' % movie_mean_ratings.max()
    movie_ratings_mean = movie_mean_ratings.mean()
    print 'The mean movie mean rating: %.2f' % movie_ratings_mean
    print 'The minimum movie mean rating: %.2f' % movie_mean_ratings.min()

    _, ax = plt.subplots(1, 1, figsize=get_fig_size())

    ax.hist(movie_mean_ratings, bins=50, alpha=0.4)

    ax.axvline(x=movie_ratings_mean, linewidth=3, color='k')
    plt.text(movie_ratings_mean + 0.05, 1150, 'mean = %.2f' % movie_ratings_mean)

    ax.set_xlabel('movie mean rating')
    ax.set_ylabel('count')
    ax.set_title('Movie mean ratings')

    plt.tight_layout()
    plt.show()


def explore_mean_movie_ratings(ratings_df):
    movie_ids = ratings_df['movieId']

    print 'How many movies? {:,}\n'.format(len(set(movie_ids)))

    num_ratings_per_movie = Counter(movie_ids).values()

    mean = np.mean(num_ratings_per_movie)
    print 'The maximum number of ratings per movie: %.0f' % np.max(num_ratings_per_movie)
    print 'The mean number of ratings per movie: %.2f' % mean
    print 'The minimum number of ratings per movie: %.0f\n' % np.min(num_ratings_per_movie)

    num_ratings_per_movie_counter = Counter(num_ratings_per_movie)

    print 'Number of movies with one rating: {:,}'.format(num_ratings_per_movie_counter[1])
    print 'Number of movies with two ratings: {:,}\n'.format(num_ratings_per_movie_counter[2])

    show_movie_mean_ratings_histogram(ratings_df)


def get_y_pred_mean_movie_ratings_model(mean_movie_ratings, mean_user_ratings, x):
    return [mean_movie_ratings.get(row['movieId'], mean_user_ratings[row['userId']]) for _, row in x.iterrows()]


def fit_mean_movie_ratings_model(ratings_df):
    train_scores = []
    test_scores = []
    n_iter = 4
    for _ in xrange(n_iter):
        train_df, test_df = train_test_split(ratings_df)

        x_train, y_train = get_xy(train_df)
        x_test, y_test = get_xy(test_df)

        mean_movie_ratings = get_mean_movie_ratings(train_df)
        mean_user_ratings = get_mean_user_ratings(train_df)

        y_train_pred = get_y_pred_mean_movie_ratings_model(mean_movie_ratings, mean_user_ratings, x_train)
        y_test_pred = get_y_pred_mean_movie_ratings_model(mean_movie_ratings, mean_user_ratings, x_test)

        train_score, test_score = get_scores(y_test, y_test_pred, y_train, y_train_pred)

        train_scores.append(train_score)
        test_scores.append(test_score)

    print 'mean train score: %.4f, std: %.4f' % (np.mean(train_scores), np.std(train_scores))
    print 'mean test score: %.4f, std: %.4f' % (np.mean(test_scores), np.std(test_scores))


def main():
    ratings_df = read_data()

    # explore_total_mean_rating(ratings_df)
    # fit_total_mean_model(ratings_df)

    # explore_mean_user_ratings(ratings_df)
    # fit_mean_user_ratings_model(ratings_df)

    explore_mean_movie_ratings(ratings_df)
    fit_mean_movie_ratings_model(ratings_df)


if __name__ == '__main__':
    main()
