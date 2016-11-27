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


def explore_data(ratings_df):
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


def fit_total_mean_model(ratings_df):
    train_df, test_df = train_test_split(ratings_df)

    print 'shapes of train and test sets: %s %s\n' % (train_df.shape, test_df.shape)

    x_train, y_train = get_xy(train_df)
    x_test, y_test = get_xy(test_df)

    y_mean = y_train.mean()

    print 'y_mean: %.2f\n' % y_mean

    y_train_pred = get_y_pred_total_mean_model(y_mean, x_train)
    y_test_pred = get_y_pred_total_mean_model(y_mean, x_test)

    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)

    print 'train score: %f' % train_score
    print 'test score: %f' % test_score


def main():
    ratings_df = read_data()

    explore_data(ratings_df)

    fit_total_mean_model(ratings_df)


if __name__ == '__main__':
    main()
