from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import f1_score
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from simpletransformers.classification import ClassificationModel
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')

RANDOM_SEED = 64

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
    texts_stem_ljubesicpandzic = []

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
            labels_fourway.append(0) # NS
        else:
            labels_subjectivity.append(1)
            if 'M' in item:
                labels_fourway.append(1) # M
            elif '+1' in item:
                labels_fourway.append(2) # +1
            else:
                labels_fourway.append(3) # -1
        if item[-1] == 's':
            item = item[:-1]
            labels_sarcasm.append(1)
        else:
            labels_sarcasm.append(0)

        if item == '+NS':
            labels_sixway.append(0)
        elif item == '-NS':
            labels_sixway.append(1)
        elif item == '+M':
            labels_sixway.append(2)
        elif item == '-M':
            labels_sixway.append(3)
        elif item == '+1':
            labels_sixway.append(4)
        elif item == '-1':
            labels_sixway.append(5)


    with open ('SentiComments.SR.orig.texts.tl.txt', 'r', encoding='utf-8') as infile:
        texts_original = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.txt', 'r', encoding='utf-8') as infile:
        texts_corrected = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.token.txt', 'r', encoding='utf-8') as infile:
        texts_normalized = [line.strip() for line in infile.readlines()]

    with open('SentiComments.SR.corr.texts.tl.repnorm.emotnorm.stem.LjubesicPandzic.txt', 'r', encoding='utf-8') as infile:
        texts_stem_ljubesicpandzic = [line.strip() for line in infile.readlines()]

    task_dict = { 'POLARITY':labels_polarity,
                  'SUBJECTIVITY':labels_subjectivity,
                  'FOUR-WAY':labels_fourway,
                  'SIX-WAY':labels_sixway
                }
    task_classes_count_dict = {'POLARITY':2, 'SUBJECTIVITY':2, 'FOUR-WAY':4, 'SIX-WAY':6}
    models_dict = { 'distilbert-base-multilingual-cased': 'distilbert',
                    'bert-base-multilingual-cased':'bert',
                    'xlm-mlm-100-1280':'xlm'
                   }

    text_variants = { 'Original Texts' : texts_original,
                      'Corrected Texts': texts_corrected,
                      'Normalized Texts': texts_normalized,
                      'Stemmed Texts': texts_stem_ljubesicpandzic
                    }

    def f1_weighted_score(y_true, y_pred, labels=None, pos_label=1, sample_weight=None, zero_division="warn"):
        return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average='weighted', sample_weight=sample_weight, zero_division=zero_division)

    parameter_dict = {}
    parameter_dict['fp16'] = False
    parameter_dict['manual_seed'] = RANDOM_SEED
    parameter_dict['overwrite_output_dir'] = True
    parameter_dict['reprocess_input_data'] = True
    parameter_dict['no_cache'] = True
    parameter_dict['save_eval_checkpoints'] = False
    parameter_dict['save_model_every_epoch'] = False
    parameter_dict['use_cached_eval_features'] = False
    parameter_dict['output_dir'] = 'G:/LRs/Transformers/outputs/'
    parameter_dict['cache_dir'] = 'G:/LRs/Transformers/cache/'
    parameter_dict['tensorboard_dir'] = 'G:/LRs/Transformers/runs/'
    parameter_dict['silent'] = True
    parameter_dict['do_lower_case'] = False

    parameter_dict['num_train_epochs'] = 1
    score_numerical_precision = '.3f'
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    for task_title, task_labels in task_dict.items():
        for model_name, model_type in models_dict.items():
            print('#################### ' + task_title +' ####################')
            print('@@@@@@@@@@ ' + model_name + ' @@@@@@@@@@')

            for text_variant, texts in text_variants.items():
                print('********** ' + text_variant + ' **********')

                X = np.array(texts)
                y = np.array(task_labels)
                f_score_per_fold = []
                results_per_fold = []

                for train_index, test_index in cv.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    train_df = pd.DataFrame(list(zip(X_train, y_train)), columns=['text', 'labels'])
                    eval_df = pd.DataFrame(list(zip(X_test, y_test)), columns=['text', 'labels'])

                    model = ClassificationModel(model_type, model_name, num_labels=task_classes_count_dict[task_title], use_cuda=True, args=parameter_dict)  # You can set class weights by using the optional weight argument
                    # Train the model
                    model.train_model(train_df, show_running_loss=False)
                    # Evaluate the model
                    result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_weighted_score)
                    results_per_fold.append(result)
                    f_score_per_fold.append(result['f1'])
                for f in f_score_per_fold:
                    print(f)
                print()
                print('CV weighted F-measure: ' + format(sum(f_score_per_fold) / 10, score_numerical_precision))
                print()
