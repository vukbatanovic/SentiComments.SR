import math
import numpy as np

movie_corpus_ig = [0.929, 0.896, 0.934, 0.892, 0.829]
movie_corpus_eg = [0.861, 0.795, 0.814, 0.750, 0.628]
movie_corpus_cg = [0.887, 0.725, 0.697, 0.679, 0.131]
book_corpus_ig = [0.935, 0.929, 0.948, 0.924, 0.931]
book_corpus_eg = [0.935, 0.889, 0.852, 0.832, 1.000]
book_corpus_cg = [0.731, 0.520, 0.570, 0.517, 0.324]

movie_corpus_speed_ig = 77.0
movie_corpus_speed_eg = 52.0
movie_corpus_speed_cg = 133.0
book_corpus_speed_ig = 87.0
book_corpus_speed_eg = 58.0
book_corpus_speed_cg = 138.0

interpretation_mapping = {}
interpretation_mapping['polarity'] = 0
interpretation_mapping['subjectivity'] = 1
interpretation_mapping['four-class'] = 2
interpretation_mapping['six-class'] = 3
interpretation_mapping['sarcasm'] = 4


def calc_delta_e(speed_cg, speed_eg):
    return (speed_cg - speed_eg) / speed_cg


def calc_delta_a(kappa_cg, kappa_eg):
    delta_a_max = ((1 - kappa_cg) - (1 - kappa_eg)) / (1 - kappa_cg)
    kappa_eg_tentative = kappa_eg
    if kappa_eg_tentative > 0.667:
        kappa_eg_tentative = 0.667
    kappa_eg_reliable = kappa_eg
    if kappa_eg_reliable > 0.8:
        kappa_eg_reliable = 0.8

    if kappa_cg <= kappa_eg:
        if kappa_cg < 0.667:
            delta_a_tentative = ((0.667 - kappa_cg) - (0.667 - kappa_eg_tentative)) / (0.667 - kappa_cg)
            delta_a_reliable = ((0.8 - kappa_cg) - (0.8 - kappa_eg_reliable)) / (0.8 - kappa_cg)
            return (delta_a_tentative + delta_a_reliable + delta_a_max)/3
        if kappa_cg < 0.8:
            delta_a_reliable = ((0.8 - kappa_cg) - (0.8 - kappa_eg_reliable)) / (0.8 - kappa_cg)
            return (delta_a_reliable + delta_a_max) /2
    return delta_a_max


def ace(kappa_cg, kappa_eg, speed_cg, speed_eg):
    return calc_delta_a(kappa_cg, kappa_eg) / calc_delta_e(speed_cg, speed_eg)


mappings = ['polarity', 'subjectivity', 'four-class', 'six-class', 'sarcasm']
for mapping in mappings:
    print('Interpretation: ' + mapping)
    movie_kappa_ig = movie_corpus_ig[interpretation_mapping[mapping]]
    movie_kappa_eg = movie_corpus_eg[interpretation_mapping[mapping]]
    movie_kappa_cg = movie_corpus_cg[interpretation_mapping[mapping]]
    book_kappa_ig = book_corpus_ig[interpretation_mapping[mapping]]
    book_kappa_eg = book_corpus_eg[interpretation_mapping[mapping]]
    book_kappa_cg = book_corpus_cg[interpretation_mapping[mapping]]
    print('Movie corpus IG vs CG: ' + format(
        ace(movie_kappa_cg, movie_kappa_ig, movie_corpus_speed_cg, movie_corpus_speed_ig), '.3f'))
    print('Movie corpus EG vs CG: ' + format(
        ace(movie_kappa_cg, movie_kappa_eg, movie_corpus_speed_cg, movie_corpus_speed_eg), '.3f'))
    print(
        'Book corpus IG vs CG: ' + format(ace(book_kappa_cg, book_kappa_ig, book_corpus_speed_cg, book_corpus_speed_ig),
                                          '.3f'))
    print(
        'Book corpus EG vs CG: ' + format(ace(book_kappa_cg, book_kappa_eg, book_corpus_speed_cg, book_corpus_speed_eg),
                                          '.3f'))
    print()
