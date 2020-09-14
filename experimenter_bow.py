from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import time
import numpy as np
from nbsvm import NBSVM

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

RANDOM_SEED = 64
def negate_processed_texts(negated_texts, processed_texts):
    negated_processed_texts = []
    for negated_text, processed_text in zip(negated_texts, processed_texts):
        negated_text_tokens = negated_text.split()
        processed_text_tokens = processed_text.split()
        if len(negated_text_tokens) != len(processed_text_tokens):
            print("ERROR")
            exit(-1)
        else:
            negated_processed_text = []
            for i in range(0, len(negated_text_tokens)):
                if negated_text_tokens[i][0:3] == 'NE_':
                    negated_processed_text.append('NE_' + processed_text_tokens[i])
                else:
                    negated_processed_text.append(processed_text_tokens[i])
            negated_processed_texts.append(' '.join(negated_processed_text))
    return negated_processed_texts

if __name__ == '__main__':
    start_time = time.time()
    print('Current time: ' + time.ctime() + '\n')
    np.random.seed(RANDOM_SEED)

    ##### DATA READING #####################################################################################################

    labels_full = []
    labels_polarity = []
    labels_subjectivity = []
    labels_fourway = []
    labels_sixway = []
    labels_sarcasm = []

    texts_original = []
    texts_corrected = []
    texts_normalized = []
    texts_stem_ksgreedy = []
    texts_stem_ksoptimal = []
    texts_stem_milosevic = []
    texts_stem_ljubesicpandzic = []
    texts_lemma_btagger_suffix = []
    texts_lemma_btagger_prefixsuffix = []
    texts_lemma_cst = []
    texts_lemma_reldi = []
    texts_normalized_negated_1 = []
    texts_normalized_negated_2 = []
    texts_normalized_negated_3 = []
    texts_normalized_negated_5 = []
    texts_normalized_negated_full = []

    with open ('SentiComments.SR.orig.txt', 'r', encoding='utf-8') as infile:
        labels_full = [line.strip().split('\t')[0] for line in infile.readlines()]

    labels_polarity = [item[0] for item in labels_full]
    for i in range(0, len(labels_polarity)):
        if labels_polarity[i] == '+':
            labels_polarity[i] = 1
        else:
            labels_polarity[i] = 0

    for item in labels_full:
        if 'NS' in item:
            labels_subjectivity.append(0)
            labels_fourway.append('NS')
        else:
            labels_subjectivity.append(1)
            if 'M' in item:
                labels_fourway.append('M')
            elif '+1' in item:
                labels_fourway.append('+1')
            else:
                labels_fourway.append('-1')
        if item[-1] == 's':
            item = item[:-1]
            labels_sarcasm.append(1)
        else:
            labels_sarcasm.append(0)
        labels_sixway.append(item)


    with open ('SentiComments.SR.orig.texts.tl.txt', 'r', encoding='utf-8') as infile:
        texts_original = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.txt', 'r', encoding='utf-8') as infile:
        texts_corrected = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.txt', 'r', encoding='utf-8') as infile:
        texts_normalized = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.stem.KeseljSipkaGreedy.txt', 'r', encoding='utf-8') as infile:
        texts_stem_ksgreedy = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.stem.KeseljSipkaOptimal.txt', 'r', encoding='utf-8') as infile:
        texts_stem_ksoptimal = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.stem.Milosevic.txt', 'r', encoding='utf-8') as infile:
        texts_stem_milosevic = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.stem.LjubesicPandzic.txt', 'r', encoding='utf-8') as infile:
        texts_stem_ljubesicpandzic = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.lemma.BTagger.suffix.txt', 'r', encoding='utf-8') as infile:
        texts_lemma_btagger_suffix = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.lemma.BTagger.prefixsuffix.txt', 'r', encoding='utf-8') as infile:
        texts_lemma_btagger_prefixsuffix = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.lemma.CST.txt', 'r', encoding='utf-8') as infile:
        texts_lemma_cst = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.lemma.ReLDI.txt', 'r', encoding='utf-8') as infile:
        texts_lemma_reldi = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.negate1.txt', 'r', encoding='utf-8') as infile:
        texts_normalized_negated_1 = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.negate2.txt', 'r', encoding='utf-8') as infile:
        texts_normalized_negated_2 = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.negate3.txt', 'r', encoding='utf-8') as infile:
        texts_normalized_negated_3 = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.negate5.txt', 'r', encoding='utf-8') as infile:
        texts_normalized_negated_5 = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.negatefull.txt', 'r', encoding='utf-8') as infile:
        texts_normalized_negated_full = [line.strip() for line in infile.readlines()]


    task_dict = {'POLARITY':labels_polarity, 'SUBJECTIVITY':labels_subjectivity, 'FOUR-WAY':labels_fourway, 'SIX-WAY':labels_sixway, 'SARCASM': labels_sarcasm}

    ########################################################################################################################

    scoring_measure_dict = {'POLARITY': 'f1_weighted',
                            'SUBJECTIVITY': 'f1_weighted',
                            'FOUR-WAY':'f1_weighted',
                            'SIX-WAY':'f1_weighted',
                            'SARCASM': make_scorer(f1_score, pos_label=1),
                            }

    score_numerical_precision = '.3f'

    clf_baseline = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', DummyClassifier(strategy='prior', random_state=RANDOM_SEED))])
    clf_bow_count_mnb = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', MultinomialNB())])
    clf_bow_count_cnb = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', ComplementNB())])
    clf_bow_count_lr = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'))])
    clf_bow_count_svm = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', LinearSVC(random_state=RANDOM_SEED, max_iter=1000000))])
    clf_bow_count_nbsvm = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1))), ('clf', NBSVM(random_state=RANDOM_SEED, max_iter=1000000))])

    clf_bow_count_mnb_bigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 2))), ('clf', MultinomialNB())])
    clf_bow_count_cnb_bigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 2))), ('clf', ComplementNB())])
    clf_bow_count_lr_bigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 2))), ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'))])
    clf_bow_count_svm_bigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 2))), ('clf', LinearSVC(random_state=RANDOM_SEED, max_iter=1000000))])
    clf_bow_count_nbsvm_bigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 2))), ('clf', NBSVM(random_state=RANDOM_SEED, max_iter=1000000))])

    clf_bow_count_mnb_trigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 3))), ('clf', MultinomialNB())])
    clf_bow_count_cnb_trigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 3))), ('clf', ComplementNB())])
    clf_bow_count_lr_trigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 3))), ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'))])
    clf_bow_count_svm_trigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 3))), ('clf', LinearSVC(random_state=RANDOM_SEED, max_iter=1000000))])
    clf_bow_count_nbsvm_trigram = Pipeline([('countvectorizer', CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 3))), ('clf', NBSVM(random_state=RANDOM_SEED, max_iter=1000000))])


    clf_bow_tfidf_mnb = Pipeline([('tfidfvectorizer', TfidfVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1), sublinear_tf=True, norm=None)), ('clf', MultinomialNB())])
    clf_bow_tfidf_cnb = Pipeline([('tfidfvectorizer', TfidfVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1), sublinear_tf=True, norm=None)), ('clf', ComplementNB())])
    clf_bow_tfidf_lr = Pipeline([('tfidfvectorizer', TfidfVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1), sublinear_tf=True)), ('clf', LogisticRegression(random_state=RANDOM_SEED, solver='liblinear'))])
    clf_bow_tfidf_svm = Pipeline([('tfidfvectorizer', TfidfVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1), sublinear_tf=True)), ('clf', LinearSVC(random_state=RANDOM_SEED, max_iter=1000000))])
    clf_bow_tfidf_nbsvm = Pipeline([('tfidfvectorizer', TfidfVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1), sublinear_tf=True)), ('clf', NBSVM(random_state=RANDOM_SEED, max_iter=1000000))])


    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    p_grid_lr_svm = {'clf__C': [0.01, 0.1, 1.0, 10, 100]}
    p_grid_nbsvm = {'clf__C': [0.01, 0.1, 1.0, 10, 100], 'clf__beta':[0.25, 0.5]}

    def eval_loop(name, classifier, texts, labels, scoring_measure):
        scores_bow = cross_val_score(classifier, texts, labels, cv=outer_cv, n_jobs=-1, scoring=scoring_measure)
        print(name + format(scores_bow.mean(), score_numerical_precision))

    for task_title, task_labels in task_dict.items():
        scoring_measure = scoring_measure_dict[task_title]

        gs_clf_bow_count_lr = GridSearchCV(estimator=clf_bow_count_lr, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_lr_bigram = GridSearchCV(estimator=clf_bow_count_lr_bigram, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_lr_trigram = GridSearchCV(estimator=clf_bow_count_lr_trigram, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)

        gs_clf_bow_tfidf_lr = GridSearchCV(estimator=clf_bow_tfidf_lr, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)

        gs_clf_bow_count_svm = GridSearchCV(estimator=clf_bow_count_svm, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_svm_bigram = GridSearchCV(estimator=clf_bow_count_svm_bigram, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_svm_trigram = GridSearchCV(estimator=clf_bow_count_svm_trigram, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)

        gs_clf_bow_tfidf_svm = GridSearchCV(estimator=clf_bow_tfidf_svm, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)

        gs_clf_bow_count_nbsvm = GridSearchCV(estimator=clf_bow_count_nbsvm, param_grid=p_grid_nbsvm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_nbsvm_bigram = GridSearchCV(estimator=clf_bow_count_nbsvm_bigram, param_grid=p_grid_nbsvm, cv=inner_cv, scoring=scoring_measure)
        gs_clf_bow_count_nbsvm_trigram = GridSearchCV(estimator=clf_bow_count_nbsvm_trigram, param_grid=p_grid_nbsvm, cv=inner_cv, scoring=scoring_measure)

        gs_clf_bow_tfidf_nbsvm = GridSearchCV(estimator=clf_bow_tfidf_nbsvm, param_grid=p_grid_nbsvm, cv=inner_cv, scoring=scoring_measure)


        classifiers = {'Baseline': {'count': clf_baseline},
                       'Multinomial Naive Bayes': {'count': clf_bow_count_mnb, 'count2': clf_bow_count_mnb_bigram, 'count3': clf_bow_count_mnb_trigram, 'tfidf':clf_bow_tfidf_mnb},
                       'Complement Naive Bayes': {'count': clf_bow_count_cnb, 'count2': clf_bow_count_cnb_bigram, 'count3': clf_bow_count_cnb_trigram, 'tfidf':clf_bow_tfidf_cnb},
                       'Logistic Regression': {'count': gs_clf_bow_count_lr, 'count2':gs_clf_bow_count_lr_bigram, 'count3':gs_clf_bow_count_lr_trigram, 'tfidf': gs_clf_bow_tfidf_lr},
                       'Linear SVM': {'count': gs_clf_bow_count_svm, 'count2': gs_clf_bow_count_svm_bigram, 'count3':gs_clf_bow_count_svm_trigram, 'tfidf': gs_clf_bow_tfidf_svm},
                      }

        if len(np.unique(task_labels)) == 2:
            classifiers['NBSVM'] = {'count': gs_clf_bow_count_nbsvm, 'count2': gs_clf_bow_count_nbsvm_bigram, 'count3':gs_clf_bow_count_nbsvm_trigram, 'tfidf': gs_clf_bow_tfidf_nbsvm}

        negated_stemmed_texts_1 = negate_processed_texts(texts_normalized_negated_1, texts_stem_ljubesicpandzic)
        negated_stemmed_texts_2 = negate_processed_texts(texts_normalized_negated_2, texts_stem_ljubesicpandzic)
        negated_stemmed_texts_3 = negate_processed_texts(texts_normalized_negated_3, texts_stem_ljubesicpandzic)
        negated_stemmed_texts_5 = negate_processed_texts(texts_normalized_negated_5, texts_stem_ljubesicpandzic)
        negated_stemmed_texts_full = negate_processed_texts(texts_normalized_negated_full, texts_stem_ljubesicpandzic)

        for classifier_name, clf in classifiers.items():
            print('#################### ' + task_title +' ####################')
            print('@@@@@@@@@@ ' + classifier_name + ' @@@@@@@@@@')
            eval_loop('Bag-of-words - original texts: ', clf['count'], texts_original, task_labels, scoring_measure)
            if classifier_name == 'Baseline':
                continue
            eval_loop('Bag-of-words - corrected texts: ', clf['count'], texts_corrected, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts: ', clf['count'], texts_normalized, task_labels, scoring_measure)

            print()

            eval_loop('Bag-of-words - normalized texts - Stemmer - K&S Optimal: ', clf['count'], texts_stem_ksoptimal, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Stemmer - K&S Greedy: ', clf['count'], texts_stem_ksgreedy, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Stemmer - Milošević: ', clf['count'], texts_stem_milosevic, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Stemmer - Ljubešić & Pandžić: ', clf['count'], texts_stem_ljubesicpandzic, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Lemmatizer - BTagger suffix: ', clf['count'], texts_lemma_btagger_suffix, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Lemmatizer - BTagger prefix+suffix: ', clf['count'], texts_lemma_btagger_prefixsuffix, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Lemmatizer - CST: ', clf['count'], texts_lemma_cst, task_labels, scoring_measure)
            eval_loop('Bag-of-words - normalized texts - Lemmatizer - ReLDI: ', clf['count'], texts_lemma_reldi, task_labels, scoring_measure)
            print()

            if task_title != 'SARCASM':
                eval_loop('Bag-of-words - normalized texts = Stemmer - Ljubešić & Pandžić - negated 1: ', clf['count'], negated_stemmed_texts_1, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts = Stemmer - Ljubešić & Pandžić - negated 2: ', clf['count'], negated_stemmed_texts_2, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts = Stemmer - Ljubešić & Pandžić - negated 3: ', clf['count'], negated_stemmed_texts_3, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts = Stemmer - Ljubešić & Pandžić - negated 5: ', clf['count'], negated_stemmed_texts_5, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts = Stemmer - Ljubešić & Pandžić - negated full: ', clf['count'], negated_stemmed_texts_full, task_labels, scoring_measure)
            else:
                eval_loop('Bag-of-words - normalized texts negated 1: ', clf['count'], texts_normalized_negated_1, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts negated 2: ', clf['count'], texts_normalized_negated_2, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts negated 3: ', clf['count'], texts_normalized_negated_3, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts negated 5: ', clf['count'], texts_normalized_negated_5, task_labels, scoring_measure)
                eval_loop('Bag-of-words - normalized texts negated full: ', clf['count'], texts_normalized_negated_full, task_labels, scoring_measure)

            print()

            if task_title == 'POLARITY':
                eval_loop('Bag-of-words - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2 - TFIDF: ', clf['tfidf'], negated_stemmed_texts_2, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2: ', clf['count2'], negated_stemmed_texts_2, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2+3 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2: ', clf['count3'], negated_stemmed_texts_2, task_labels, scoring_measure)

            elif task_title == 'SUBJECTIVITY':
                eval_loop('Bag-of-words - normalized texts - Stemmer - Ljubešić & Pandžić - TFIDF: ', clf['tfidf'], texts_stem_ljubesicpandzic, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić: ', clf['count2'], texts_stem_ljubesicpandzic, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2+3 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić: ', clf['count3'], texts_stem_ljubesicpandzic, task_labels, scoring_measure)

            elif task_title == 'FOUR-WAY':
                eval_loop('Bag-of-words - normalized texts - Stemmer - Ljubešić & Pandžić - negated 1 - TFIDF: ', clf['tfidf'], negated_stemmed_texts_1, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 1: ', clf['count2'], negated_stemmed_texts_1, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2+3 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 1: ', clf['count3'], negated_stemmed_texts_1, task_labels, scoring_measure)

            elif task_title == 'SIX-WAY':
                eval_loop('Bag-of-words - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2 - TFIDF: ', clf['tfidf'], negated_stemmed_texts_2, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2: ', clf['count2'], negated_stemmed_texts_2, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2+3 n-grams - normalized texts - Stemmer - Ljubešić & Pandžić - negated 2: ', clf['count3'], negated_stemmed_texts_2, task_labels, scoring_measure)

            elif task_title == 'SARCASM':
                eval_loop('Bag-of-words - normalized texts - TFIDF: ', clf['tfidf'], texts_normalized, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2 n-grams - normalized texts: ', clf['count2'], texts_normalized, task_labels, scoring_measure)
                eval_loop('Bag-of-1+2+3 n-grams - normalized texts: ', clf['count3'], texts_normalized, task_labels, scoring_measure)

            print()
        print()

    end_time = time.time()
    print('Total time elapsed: {0:f} seconds.\n'.format(end_time - start_time))
