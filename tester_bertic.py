import numpy as np
from simpletransformers.classification import ClassificationModel
from transformers import logging

# Set the logging verbosity
logging.set_verbosity_error()

# Setting a random seed in order to ensure result reproducibility
RANDOM_SEED = 64
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':

    # These texts are provided as an illustration and can freely be changed/replaced with your own examples.
    test_texts = ["Ovo je jako dobar primer!",
                  "Ništa mi se ne sviđa.",
                  "Sutra for će biti toplo.",
                  "Da li je ovo stvarno istina?",
                  "Omanuo si u proceni, meni je sasvim ok.",
                  "Ima kvalitetnu reklamu, ali mi se ipak baš ne dopada."
                  ]

    labels_polarity = []
    labels_subjectivity = []
    labels_fourway = []
    labels_sixway = []

    # Labels/classes have to be integer values, so they have to inverse mapped for printing
    label_dict_polarity = {0:'-', 1:'+'}
    label_dict_subjectivity = {0:'Objective', 1:'Subjective'}
    label_dict_fourway = {0:'NS', 1:'M', 2:'+1', 3:'-1'}
    label_dict_sixway = {0:'+NS', 1:'-NS', 2:'+M', 3:'-M', 4:'+1', 5:'-1'}

    # Select which classification task(s) ought to be performed - comment out those which should not be considered
    task_dict = { 'POLARITY DETECTION': label_dict_polarity,
                  'SUBJECTIVITY DETECTION': label_dict_subjectivity,
                  'FOUR-WAY CLASSIFICATION': label_dict_fourway,
                  'SIX-WAY CLASSIFICATION': label_dict_sixway
                 }

    results_dict = { 'POLARITY DETECTION': labels_polarity,
                     'SUBJECTIVITY DETECTION': labels_subjectivity,
                     'FOUR-WAY CLASSIFICATION': labels_fourway,
                     'SIX-WAY CLASSIFICATION': labels_sixway
                    }

    # This dictionary states the number of classes for each sentiment classification task
    task_classes_count_dict = {'POLARITY DETECTION': 2,
                               'SUBJECTIVITY DETECTION': 2,
                               'FOUR-WAY CLASSIFICATION': 4,
                               'SIX-WAY CLASSIFICATION': 6
                               }

    # Select the models - this is done by simply stating their full name from the Hugging Face model page:
    # https://huggingface.co/models
    # For each model we need to specify the model architecture - 'bert, 'electra', 'xlm', etc.
    # Comment out the models which should not be considered
    model_type = 'electra'
    model_names_dict = { 'POLARITY DETECTION':'ICEF-NLP/bcms-bertic-senticomments-sr-polarity',
                    'SUBJECTIVITY DETECTION': 'ICEF-NLP/bcms-bertic-senticomments-sr-subjectivity',
                    'FOUR-WAY CLASSIFICATION': 'ICEF-NLP/bcms-bertic-senticomments-sr-fourway',
                    'SIX-WAY CLASSIFICATION': 'ICEF-NLP/bcms-bertic-senticomments-sr-sixway'
                   }

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

    # Reduce the output details - set to False to enable a detailed overview
    parameter_dict['silent'] = True

    # Models we consider retain text casing
    parameter_dict['do_lower_case'] = False

    print ('INPUT TEXTS')
    for test_text in test_texts:
        print(test_text)
    print()
    print('LABEL PREDICTIONS')
    print()

    # Change the use_cuda to False if you do not have GPU support
    for task_title, model_name in model_names_dict.items():
        model = ClassificationModel(model_type, model_name, num_labels=task_classes_count_dict[task_title],
                                    use_cuda=True, args=parameter_dict)

        predictions, raw_outputs = model.predict(test_texts)
        print(task_title)
        for pred in predictions:
            results_dict[task_title].append(task_dict[task_title][pred])
        print(results_dict[task_title])
        print()