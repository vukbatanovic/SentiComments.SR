import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from simpletransformers.classification import ClassificationModel
import pandas as pd
from transformers import logging

# Set the logging verbosity
logging.set_verbosity_error()

# Setting a random seed in order to ensure result reproducibility
RANDOM_SEED = 64
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':

    # Reading data and decoding the sentiment labels into classes
    labels_full = pd.read_csv('SentiComments.SR.orig.txt', sep='\t', encoding='utf-8', header=None,
                              quoting=csv.QUOTE_NONE)[0].tolist()

    texts_corrected = []
    with open('SentiComments.SR.corr.texts.tl.txt', 'r', encoding='utf-8') as infile:
        texts_corrected = [line.strip() for line in infile.readlines()]

    labels_polarity = []
    labels_subjectivity = []
    labels_fourway = []
    labels_sixway = []

    # Labels/classes have to be integer values
    label_dict_four = {'+NS': 0, '-NS': 0, '+M': 1, '-M': 1, '+1': 2, '-1': 3}
    label_dict_six = {'+NS': 0, '-NS': 1, '+M': 2, '-M': 3, '+1': 4, '-1': 5}

    for item in labels_full:
        if item[-1] == 's':  # eliminate the sarcasm marking, since we do not use it
            item = item[:-1]
        labels_polarity.append(1 if '+' in item else 0)
        labels_subjectivity.append(0 if 'NS' in item else 1)
        labels_fourway.append(label_dict_four[item])
        labels_sixway.append(label_dict_six[item])

    # Select which classification task(s) ought to be performed - comment out those which should not be considered
    task_dict = {'POLARITY DETECTION': labels_polarity,
                 'SUBJECTIVITY DETECTION': labels_subjectivity,
                 'FOUR-WAY CLASSIFICATION': labels_fourway,
                 'SIX-WAY CLASSIFICATION': labels_sixway
                 }

    # This dictionary states the number of classes for each sentiment classification task
    task_classes_count_dict = {'POLARITY DETECTION': 2,
                               'SUBJECTIVITY DETECTION': 2,
                               'FOUR-WAY CLASSIFICATION': 4,
                               'SIX-WAY CLASSIFICATION': 6}

    # Select the models - this is done by simply stating their full name from the Hugging Face model page:
    # https://huggingface.co/models
    # For each model we need to specify the model architecture - 'bert, 'electra', 'xlm', etc.
    # Comment out the models which should not be considered
    models_dict = {#'bert-base-multilingual-cased': 'bert',
                   'classla/bcms-bertic': 'electra'
                   }

    # Weighted F-measure is used as the performance metric
    # This can changed to other options, such as macro-averaging, if desired
    def f1_weighted_score(y_true, y_pred, labels=None, pos_label=1, sample_weight=None, zero_division="warn"):
        return f1_score(y_true, y_pred, labels=labels, pos_label=pos_label, average='weighted',
                        sample_weight=sample_weight, zero_division=zero_division)


    # Parameter dict contains all the hyper-parameters related to the (Simple) Transformers library
    # The full list of available hyper-parameters is available here:
    # https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
    parameter_dict = {}

    # Setting the random seed to ensure experiment reproducibility
    parameter_dict['manual_seed'] = 10

    # Disables the mixed precision training mode, since it may cause calculation issues on some configurations
    parameter_dict['fp16'] = False

    # These options force the library to create/train a new model every time the code is run,
    # which enables experiment reproducibility
    parameter_dict['overwrite_output_dir'] = True
    parameter_dict['reprocess_input_data'] = True
    parameter_dict['no_cache'] = True
    parameter_dict['no_save'] = True
    parameter_dict['save_eval_checkpoints'] = False
    parameter_dict['save_model_every_epoch'] = False
    parameter_dict['use_cached_eval_features'] = False

    # CHANGE THIS FOR YOUR LOCAL CONFIGURATION
    # Choose the working directory for the transformers models
    parameter_dict['output_dir'] = 'G:/LRs/Transformers/outputs/'
    parameter_dict['cache_dir'] = 'G:/LRs/Transformers/cache/'
    parameter_dict['tensorboard_dir'] = 'G:/LRs/Transformers/runs/'

    # Reduce the output details - set to False to enable a detailed overview of the training process
    parameter_dict['silent'] = True

    # Both models we consider retain text casing
    parameter_dict['do_lower_case'] = False

    # Experiment with increasing the number of training epochs and see how it affects the results
    parameter_dict['num_train_epochs'] = 5

    # Some other options you could explore that influence the model - the default values are given below
    # If your GPU runs out of memory, try lowering the batch size parameter
    # parameter_dict['train_batch_size'] = 8
    # parameter_dict['eval_batch_size'] = 8
    # parameter_dict['learning_rate'] = 4e-5

    score_numerical_precision = '.3f'
    CV_SPLITS = 10
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Iterate over all classification tasks and all models
    for task_title, task_labels in task_dict.items():
        for model_name, model_type in models_dict.items():
            print('#################### ' + task_title + ' ####################')
            print('@@@@@@@@@@ ' + model_name + ' @@@@@@@@@@')

            X = np.array(texts_corrected)
            y = np.array(task_labels)
            score_per_fold = []

            for train_index, test_index in cv.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                train_df = pd.DataFrame(list(zip(X_train, y_train)), columns=['text', 'labels'])
                eval_df = pd.DataFrame(list(zip(X_test, y_test)), columns=['text', 'labels'])

                # Change the use_cuda to False if you do not have GPU support
                model = ClassificationModel(model_type, model_name, num_labels=task_classes_count_dict[task_title],
                                            use_cuda=True, args=parameter_dict)

                # Train the model
                model.train_model(train_df, show_running_loss=False)

                # Evaluate the model and print the results for each CV fold
                result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_weighted_score)
                score_per_fold.append(result['f1'])
                print(result['f1'])

            # Print the final results
            print('Final CV performance: ' + format(sum(score_per_fold) / CV_SPLITS, score_numerical_precision))
            print()
