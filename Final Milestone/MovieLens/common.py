import math
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


@contextmanager
def elapsed_time(title):
    start = time.time()
    yield
    elapsed = time.time() - start
    print '%s: %.2f secs' % (title, elapsed)


def get_xy(ratings_df):
    y = ratings_df['rating']
    x = ratings_df.drop('rating', axis=1)
    return x, y


def root_mean_squared_error(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))


def show_scores_plot(k_neighbors_values, val_scores, train_scores, model_name):
    _, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.plot(k_neighbors_values, val_scores, label='validation')
    ax.plot(k_neighbors_values, train_scores, label='train')

    ax.set_xlabel('k_neighbors')
    ax.set_ylabel('$R^2$')
    ax.set_title('Test and validation scores for different k_neighbors values (%s)' % model_name)

    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def score_model(ratings_df, model_f, model_name):
    train_val_ratings_df, test_ratings_df = train_test_split(ratings_df)

    train_ratings_df, validation_ratings_df = train_test_split(train_val_ratings_df)

    best_score = -float('inf')
    best_k_neighbors = None

    model = model_f()

    model = model.fit(train_ratings_df)

    k_neighbors_values = [1, 5, 10, 20, 30, 40, 50, 60, 80, 100]

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

    model = model_f(k_neighbors=best_k_neighbors)

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

    show_scores_plot(k_neighbors_values, val_scores, train_scores, model_name=model_name)
