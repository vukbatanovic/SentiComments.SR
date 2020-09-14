from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
import time
import sys
import os
import numpy as np
from gensim.models import Word2Vec
from embedding_vectorizer import MeanEmbeddingVectorizer, MeanEmbeddingVectorizerInNegatedText
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')

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


    task_dict = {'POLARITY':labels_polarity, 'SUBJECTIVITY':labels_subjectivity, 'FOUR-WAY':labels_fourway, 'SIX-WAY':labels_sixway}

    ########################################################################################################################

    scoring_measure_dict = {'POLARITY': 'f1_weighted',
                            'SUBJECTIVITY': 'f1_weighted',
                            'FOUR-WAY':'f1_weighted',
                            'SIX-WAY':'f1_weighted'
                            }

    score_numerical_precision = '.3f'

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    p_grid_lr_svm = {'C': [0.01, 0.1, 1.0, 10, 100]}

    w2v_models = {}

    w2v_model_original = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_original] = (texts_original, 'W2V Original: ')
    w2v_model_corrected = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_corrected] = (texts_corrected, 'W2V Corrected: ')
    w2v_model_normalized = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_normalized] = (texts_normalized, 'W2V Normalized: ')

    # MORPHOLOGICAL NORMALIZATION
    w2v_model_stem_keseljsipkaoptimal = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_KeseljSipkaOptimal/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_stem_keseljsipkaoptimal] = (texts_stem_ksoptimal, 'W2V KeseljSipkaOptimal: ')
    w2v_model_stem_keseljsipkagreedy = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_KeseljSipkaGreedy/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_stem_keseljsipkagreedy] = (texts_stem_ksgreedy, 'W2V KeseljSipkaGreedy: ')
    w2v_model_stem_milosevic = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_Milosevic/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_stem_milosevic] = (texts_stem_milosevic, 'W2V Milosevic: ')
    w2v_model_stem_ljubesicpandzic = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_stem_ljubesicpandzic] = (texts_stem_ljubesicpandzic, 'W2V LjubesicPandzic: ')
    w2v_model_lemma_btaggersuffix = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Lemma_BTagger_without_prefix/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_lemma_btaggersuffix] = (texts_lemma_btagger_suffix, 'W2V BTaggerSuffix: ')
    w2v_model_lemma_btaggerprefixsuffix = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Lemma_BTagger_prefix/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_lemma_btaggerprefixsuffix] = (texts_lemma_btagger_prefixsuffix, 'W2V BTaggerPrefixSuffix: ')
    w2v_model_lemma_cst = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Lemma_CST/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_lemma_cst] = (texts_lemma_cst, 'W2V CST: ')
    w2v_model_lemma_reldi = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Lemma_ReLDI/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_lemma_reldi] = (texts_lemma_reldi, 'W2V ReLDI: ')

    # WORD EMBEDDING PARAMETERS
    w2v_model_100_5 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.100.5.skipgram.model')
    w2v_models[w2v_model_100_5] = (texts_stem_ljubesicpandzic, 'W2V 100-5 Stemmed Ljubešić-Pandžić: ')
    w2v_model_100_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.100.10.skipgram.model')
    w2v_models[w2v_model_100_10] = (texts_stem_ljubesicpandzic, 'W2V 100-10 Stemmed Ljubešić-Pandžić: ')
    w2v_model_300_5 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.300.5.skipgram.model')
    w2v_models[w2v_model_300_5] = (texts_stem_ljubesicpandzic, 'W2V 300-5 Stemmed Ljubešić-Pandžić: ')
    w2v_model_300_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.300.10.skipgram.model')
    w2v_models[w2v_model_300_10] = (texts_stem_ljubesicpandzic, 'W2V 300-10 Stemmed Ljubešić-Pandžić: ')
    w2v_model_500_5 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.500.5.skipgram.model')
    w2v_models[w2v_model_500_5] = (texts_stem_ljubesicpandzic, 'W2V 500-5 Stemmed Ljubešić-Pandžić: ')
    w2v_model_500_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.500.10.skipgram.model')
    w2v_models[w2v_model_500_10] = (texts_stem_ljubesicpandzic, 'W2V 500-10 Stemmed Ljubešić-Pandžić: ')
    w2v_model_1000_5 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.1000.5.skipgram.model')
    w2v_models[w2v_model_1000_5] = (texts_stem_ljubesicpandzic, 'W2V 1000-5 Stemmed Ljubešić-Pandžić: ')
    w2v_model_1000_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.1000.10.skipgram.model')
    w2v_models[w2v_model_1000_10] = (texts_stem_ljubesicpandzic, 'W2V 1000-10 Stemmed Ljubešić-Pandžić: ')

    # OTHER OPTIONS
    w2v_model_1000_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Stem_LjubesicPandzic/SrWac.1000.10.skipgram.model')
    w2v_models[w2v_model_1000_10] = (texts_stem_ljubesicpandzic, 'W2V 1000-10 Stemmed Ljubešić-Pandžić: ')
    w2v_model_1000_10 = Word2Vec.load('G:/LRs/Clarin LRs/sr/SrWac_WNum_Negated_Stem/SrWac.1000.10.skipgram.model')
    w2v_models[w2v_model_1000_10] = (texts_stem_ljubesicpandzic, 'W2V 1000-10 Stemmed Ljubešić-Pandžić Negated: ')

    def eval_loop(name, classifier, texts, labels, scoring_measure):
        scores_boe = cross_val_score(classifier, texts, labels, cv=outer_cv, n_jobs=-1, scoring=scoring_measure)
        print(name + format(scores_boe.mean(), score_numerical_precision))

    def task_texts_negation_settings(texts, task_label):
        task_texts_dict = {'POLARITY':negate_processed_texts(texts_normalized_negated_2, texts),
                       'SUBJECTIVITY':texts,
                       'FOUR-WAY':negate_processed_texts(texts_normalized_negated_1, texts),
                       'SIX-WAY':negate_processed_texts(texts_normalized_negated_2, texts)
                       }
        return task_texts_dict[task_label]

    for task_title, task_labels in task_dict.items():
        scoring_measure = scoring_measure_dict[task_title]

        for w2v_model, texts in w2v_models.items():
            print('#################### ' + task_title +' ####################')
            print('@@@@@@@@@@ ' + 'classifier_name' + ' @@@@@@@@@@')
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

            word2vec_vectorizer = MeanEmbeddingVectorizer(w2v_model)
            # word2vec_vectorizer = MeanEmbeddingVectorizer(w2v_model, bow_vectorizer=CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1)))
            # word2vec_vectorizer = MeanEmbeddingVectorizerInNegatedText(w2v_model, bow_vectorizer=CountVectorizer(token_pattern=r"\S+", lowercase=True, ngram_range=(1, 1)))

            comment_vectors = word2vec_vectorizer.fit_transform(texts[0], task_labels)
            # comment_vectors = word2vec_vectorizer.fit_transform(task_texts_negation_settings(texts[0], task_title), task_labels)

            clf_word2vec = LinearSVC(random_state=RANDOM_SEED)
            gs_clf_word2vec = GridSearchCV(estimator=clf_word2vec, param_grid=p_grid_lr_svm, cv=inner_cv, scoring=scoring_measure)
            eval_loop(texts[1], gs_clf_word2vec, comment_vectors, task_labels, scoring_measure)
            print()

        print()

    end_time = time.time()
    print('Total time elapsed: {0:f} seconds.\n'.format(end_time - start_time))
