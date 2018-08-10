import pandas as pd
from feature_extraction import TextToVec
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import f1_score
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")
import logging_util
import os.path as path
# Load data

logdir = "../logging"
if not os.path.exists(logdir):
    os.makedirs(logdir)


data_path = '/Users/sli/Projects/data'
search_path = data_path + '/hyperparameter_search/mental_health_forum_simple_clf'
dataset_path = data_path + '/mental_health_forum_data'
model_path = search_path + '/save_model'
logger = logging_util.logger("Hyper_search_simple_model", logging_folder=logdir)

# Prepare data and label
cat_num = 7

def data_prep(df):
    data = list(df.dialog)
    label_encode = LabelEncoder()
    output = dict()
    output['data'] = data
    output['encoded_label'] = label_encode.fit_transform(df.label)
    output['binary_label'] = label_binarize(y=output['encoded_label'], classes=np.arange(cat_num))
    return output


def training(train_data, dev_data, param):

    text_to_vec = TextToVec(**param)
    text_to_vec.fit(train_data['data'] + dev_data['data'])
    train_vec = text_to_vec.transform(train_data['data'])
    dev_vec = text_to_vec.transform(dev_data['data'])
    logger.info(f"train vec size:{train_vec.shape}, dev vec size:{dev_vec.shape}")

    # Fit model
    if param['classifier'] == 'MultinomialNB':
        clf = MultinomialNB()
    elif param['classifier'] == 'LDA':
        clf = LinearDiscriminantAnalysis()
    else:
        clf = svm.LinearSVC()

    if param['multiclass'] == 'OnevsOne':
        model = OneVsOneClassifier(clf)
    else:
        model = OneVsRestClassifier(clf)

    if param['classifier'] == 'LinearSVM' or param['multiclass'] == 'OnevsOne':
        logger.info(f'Fitting model: {param}')
        model = model.fit(train_vec, train_data['encoded_label'])
        train_prediction = model.predict(train_vec)
        dev_prediction = model.predict(dev_vec)
    else:
        logger.info(f'Fitting model: {param}')
        model = model.fit(train_vec, train_data['binary_label'])
        train_prediction = np.argmax(model.predict(train_vec), axis=1)
        dev_prediction = np.argmax(model.predict(dev_vec), axis=1)

    return train_prediction, dev_prediction, train_vec.shape, dev_vec.shape, model


def save_model(model, filename):

    with open(filename, 'wb') as f:
        joblib.dump(model, f)


def f1_class(target, prediction):
    f1_array = f1_score(target, prediction, labels=np.arange(7), average=None)
    return np.round(f1_array, 2)

def load_data(params):
    train_df = pd.read_csv(os.path.join(*[dataset_path, params['tokenizer'] + '_train.csv']))
    dev_df = pd.read_csv(os.path.join(*[dataset_path, params['tokenizer'] + '_dev.csv']))
    train_data = data_prep(train_df)
    dev_data = data_prep(dev_df)
    return train_data, dev_data

def list_to_str(list):
    return ' '.join([str(c) for c in list])

# Save results
def main(index, params, timestamp):

    try:
        hyperpara_search = pd.read_csv(search_path+f'/result{timestamp}.csv', usecols=np.arange(1, 11))
    except:
        columns = list(params.keys()) + ['train_f1', 'dev_f1']
        hyperpara_search = pd.DataFrame(data=None, columns=columns)

    # # generate vector data
    train_data, dev_data = load_data(params)
    train_pred, dev_pred, train_shape, dev_shape, model = training(train_data, dev_data, params)
    save_model(model, os.path.join(model_path, str(timestamp)+'_'+str(index)+'.sav'))
    params['train_size'] = train_shape
    params['dev_size'] = dev_shape
    train_f1 = f1_class(train_data['encoded_label'], train_pred)
    params['train_f1'] = list_to_str(train_f1)
    dev_f1 = f1_class(dev_data['encoded_label'], dev_pred)
    params['dev_f1'] = list_to_str(dev_f1)
    single_entry = pd.Series(params)

    hyperpara_search = hyperpara_search.append(single_entry, ignore_index=True)

    hyperpara_search.drop('para_index', axis=1, inplace=True)
    logger.info(f'saving csv up to index:{index}...')
    hyperpara_search.to_csv(search_path+f'/result{timestamp}.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=int)
    args = parser.parse_args()

    param_filename = os.path.join(*[search_path, 'param.json'])

    with open(param_filename, 'r') as f:
        param = json.load(f)

    logger.info(f"Initializing timestamp = {args.timestamp}...")

    main(param['para_index'], param, args.timestamp)
