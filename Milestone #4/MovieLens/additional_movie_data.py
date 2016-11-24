# encoding=utf8

import csv
import sys
import time
from collections import namedtuple

import pandas as pd
from imdbpie import Imdb

reload(sys)
sys.setdefaultencoding('utf8')

Dataset = namedtuple('Dataset', ['movies_df', 'links_df'])

imdb_movie_field_names = [
    'imdbId',
    'title',
    'type',
    'year',
    'tagline',
    'rating',
    'certification',
    'genres',
    'num_votes',
    'runtime',
    'release_date',
    'directors_summary',
    'creators',
    'cast_summary',
    'writers_summary',
]

ImdbMovie = namedtuple('ImdbMovie', imdb_movie_field_names)


def explore_imdb():
    imdb = Imdb(anonymize=True)

    title = imdb.get_title_by_id("tt0468569")

    print 'title: %s' % title.title
    print 'type: %s' % title.type
    print 'year: %s' % title.year
    print 'tagline: %s' % title.tagline
    print 'rating: %s' % title.rating
    print 'certification: %s' % title.certification
    print 'genres: %s' % title.genres
    print 'num_votes: %s' % title.votes
    print 'runtime: %s' % title.runtime
    print 'release_date: %s' % title.release_date
    print 'directors_summary: %s' % title.directors_summary
    print 'creators: %s' % title.creators
    print 'cast_summary: %s' % title.cast_summary
    print 'writers_summary: %s' % title.writers_summary


def read_data():
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    links_df = pd.read_csv('ml-latest-small/links.csv')
    return Dataset(movies_df, links_df)


def get_merged_movies_df(dataset):
    movies_df = dataset.movies_df
    links_df = dataset.links_df

    merged_movies_df = movies_df.merge(links_df)

    return merged_movies_df


def convert_person_list(person_list):
    return [person.name for person in person_list]


def retrieve_imdb_info(movies_df):
    imdb = Imdb(anonymize=True)

    # movies_df = movies_df.head()

    start = time.time()

    start_row = movies_df.loc[movies_df['imdbId'] == 2267968].iloc[0]
    start_row_index = start_row.name
    indices = movies_df.index[start_row_index + 1:]

    # indices = indices[:2]

    conversion_dict = {
        81454: 79285,
        56600: 54333,
        266860: 235679,
        250305: 157472,
        313487: 287635,
        290538: 270288,
        282674: 169102,
        1522863: 1493943,
        3630276: 2121382,
        5248968: 5235348,
    }

    with open('imdb_movies.csv', 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')

        for index in indices:
            row = movies_df.iloc[index]
            imdb_id = row['imdbId']

            imdb_id = conversion_dict.get(imdb_id, imdb_id)

            imdb_id_str = 'tt%07d' % imdb_id

            # print imdb_id, imdb_id_str

            imdb_title = imdb.get_title_by_id(imdb_id_str)

            elapsed_time = int(round(time.time() - start, 0))
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            print '%s %d:%02d %s' % (index, minutes, seconds, imdb_title)

            imdb_movie = ImdbMovie(imdb_id,
                                   imdb_title.title,
                                   imdb_title.type,
                                   imdb_title.year,
                                   imdb_title.tagline,
                                   imdb_title.rating,
                                   imdb_title.certification,
                                   imdb_title.genres,
                                   imdb_title.votes,
                                   imdb_title.runtime,
                                   imdb_title.release_date,
                                   convert_person_list(imdb_title.directors_summary),
                                   convert_person_list(imdb_title.creators),
                                   convert_person_list(imdb_title.cast_summary),
                                   convert_person_list(imdb_title.writers_summary),
                                   )

            csv_writer.writerow(imdb_movie)


def main():
    # explore_imdb()

    dataset = read_data()
    merged_movies_df = get_merged_movies_df(dataset)

    retrieve_imdb_info(merged_movies_df)


if __name__ == '__main__':
    main()
