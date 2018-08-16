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
from collections import Counter
warnings.filterwarnings("ignore")
import logging_util
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os.path as path
# Load data

logdir = "../logging"
if not os.path.exists(logdir):
    os.makedirs(logdir)


data_path = '/Users/sli/Projects/data'
search_path = data_path + '/hyperparameter_search/mental_health_forum_simple_clf'
dataset_path = data_path + '/mental_health_forum_data'
logger = logging_util.logger("Hyper_search_simple_model", logging_folder=logdir)

# Prepare data and label

def data_prep(df, params, if_resample=False):
    """
    Convert data from dataframe format into tensor of input and target
    :param df: dataframe containing disorder name, dialog
    :param params: parameter for data processing
    :param if_resample: whether to perform resampling to balance the sample size
    :return:
    output: dictionary containing data, encoded_label, binary_label
    label_encode: LabelEncoder() object for inverse fitting
    """

    if if_resample and (params['balanced'] in ['Bootstrap', 'Handsample']):
        if params['balanced'] == 'Bootstrap':
            df = resample(df=df, balance=params['balanced'], nclass=params['classnum'])
        elif params['balanced'] == 'Handsample':
            df = resample(df=df, balance=params['balanced'], nclass=params['classnum'])

    if params['classnum'] == 6:
        df.drop(df[df['label']=='PTSD'].index, axis=0, inplace=True)

    data = list(df.dialog)
    label_encode = LabelEncoder()
    output = dict()
    output['data'] = data
    output['encoded_label'] = label_encode.fit_transform(df.label)
    output['binary_label'] = label_binarize(y=output['encoded_label'], classes=np.arange(params['classnum']))
    return output, label_encode


def bootstrap(df, nclass, if_new=False):
    """
    Perform boot strapping using sampling with replacement
    :param df: dataframe to perform bootstrap from
    :param if_new: whether to generate new dataframe (downsample) or append on the original dataframe (upsample)
    :return: data frame after resampling
    """
    ori_size = Counter(df.label)
    logger.info(f'class info before resampling: {ori_size.values()}')
    ori_size_list = list(ori_size.values())

    if if_new:
        df_new = pd.DataFrame(data=None, columns=df.columns)
        target_size = min(ori_size_list)
    else:
        target_size = max(ori_size_list)
        df_new = df.copy()

    for i in range(nclass):
        name = list(ori_size.keys())[i]
        name_index = np.array(df[df.label == name].index)
        if target_size < ori_size_list[i]:
            sample_size = target_size
        elif target_size > ori_size_list[i]:
            sample_size = target_size - ori_size_list[i]
        else:
            if if_new:
                sample_size = target_size
            else:
                sample_size = 0

        np.random.seed(i)
        boostrap_sample = np.random.randint(0, ori_size_list[i], sample_size)
        df_new = df_new.append(df.iloc[name_index[boostrap_sample]], ignore_index=True)
    logger.info(f'class info after resampling: {Counter(df_new.label).values()}')
    return df_new


def resample(X_train=None, y_train=None, df=None, balance=None, nclass=None):
    """
    Perform resampling based on chosen method
    :param X_train:  tensor of training data (size, dimension)
    :param y_train:  tensor of training target (size, 1)
    :param df: dataframe to resample from
    :param balance: type of resample to perform
    :return: resampled data
    """

    if balance == 'SMOTE':
        X_train, y_train = SMOTE().fit_sample(X_train, y_train)
        logger.info(f'Using {balance} oversampling')
    elif balance == 'RandomUnderSampler':
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_sample(X_train, y_train)
        logger.info(f'Using {balance} oversampling')
    elif balance == 'Bootstrap':
        logger.info(f'Using {balance} oversampling')
        df = bootstrap(df, nclass)
        return df
    elif balance == 'Handsample':
        logger.info(f'Using {balance} oversampling')
        df = bootstrap(df, nclass, if_new=True)
        return df

    return X_train, y_train


def training(train_data, dev_data, param):
    """
    Train the model on train_data and generate prediction on train_data, dev_data
    :param train_data: dictionary containing data, encoded label and binary label
    :param dev_data: dictionary containing data, encoded label and binary label
    :param param: parameter for training
    :return:
    train_prediction: prediction of training data
    dev_prediction: prediction of development data
    train_vec.shape: shape of training vector (sample size, feature size)
    dev_vec.shape: shape of development vector (sample size, feature size)
    model: trained classifier
    word_vec_map: learned tfidf/count vectorizer
    """
    text_to_vec = TextToVec(**param)

    # Fit with both train and dev data
    text_to_vec.fit(train_data['data'] + dev_data['data'])
    word_vec_map = text_to_vec.vectorizer.get_feature_names()
    train_vec = text_to_vec.transform(train_data['data'])
    dev_vec = text_to_vec.transform(dev_data['data'])
    logger.info(f"train vec size:{train_vec.shape}, dev vec size:{dev_vec.shape}")

    # # apply weights on tfidf based on whether the word appear in multiple classes
    # tt_occ = Counter(train_data['encoded_label'])
    # weight_list = []
    # for i in range(train_vec.shape[1]):  # For every feature
    #     occ = Counter(train_data['encoded_label'][train_vec[:, i] > 0.0])
    #     for key, value in occ.items():
    #         occ[key] = value/tt_occ[key]
    #     weight_list.append(np.std(list(occ.values()))/0.35)
    # weight = np.array(weight_list).reshape(1, -1)
    # weight = weight/np.max(weight)
    # train_vec = np.multiply(train_vec, weight)

    # Perform oversampling on training data
    if param['balanced'] not in ['Bootstrap', 'Handsample']:
        logger.info(f"class info before resampling: {sorted(Counter(train_data['encoded_label']).items())}")
        train_vec, train_data['encoded_label'] = resample(X_train=train_vec, y_train=train_data['encoded_label'], balance=param['balanced'])
        logger.info(f"class info after resampling:{sorted(Counter(train_data['encoded_label']).items())}")

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


    return train_prediction, dev_prediction, train_vec.shape, dev_vec.shape, model, word_vec_map


def save_model(model, filename):
    """
    Saved the trained classifier
    """
    with open(filename, 'wb') as f:
        joblib.dump(model, f)


def f1_class(target, prediction, params):
    """
    Return the class-wise f1 score
    """
    f1_array = f1_score(target, prediction, labels=np.arange(params['classnum']), average=None)
    return np.round(f1_array, 4)


def load_data(params):
    """
    Load data from disk corresponding to the stemming and tokenizer method"
    """
    train_df = pd.read_csv(os.path.join(*[dataset_path, params['tokenizer'] + '_train_' + params['stemming'] + '.csv']))
    dev_df = pd.read_csv(os.path.join(*[dataset_path, params['tokenizer'] + '_dev_' + params['stemming'] + '.csv']))
    train_data, label_encode = data_prep(train_df, params, if_resample=True)
    dev_data, _ = data_prep(dev_df, params)
    return train_data, dev_data, label_encode


def list_to_str(list):
    return ' '.join([str(c) for c in list])


def filtered_param(tokenizer=None, stemming=None, tfidf=None, binary=None, balanced=None, rate=None, classifier=None,
                   multiclass=None, ngram_range=None, classnum=None, **kwargs):
    """
    Convert the format of input parameter to the ones easy to use
    """
    params = dict()
    params['stemming'] = stemming
    params['tokenizer'] = tokenizer
    params['tfidf'] = tfidf
    params['binary'] = binary
    params['balanced'] = balanced
    params['min_rate'] = rate[0]
    params['max_rate'] = rate[1]
    params['classifier'] = classifier
    params['multiclass'] = multiclass
    params['ngram_range'] = ngram_range[1]
    params['classnum'] = classnum

    return params


def populated(params, hyperpara_search):
    """
    Check to see if the parameter has been populated
    :param params: parameter dictionary to check
    :param hyperpara_search: data frame from original populated result
    :return:
    """

    params = filtered_param(**params)

    try:
        filter_result = np.logical_and.reduce([hyperpara_search[a] == b for a, b in params.items()])
    except:
        return False, params

    result = np.logical_or.reduce(filter_result)

    return result, params


def main(params, timestamp):

    # ----------Fetch existing df or build empty one
    try:
        hyperpara_search = pd.read_csv(search_path + f'/result{timestamp}.csv', usecols=np.arange(1, 18))
    except:
        columns = list(params.keys()) + ['train_f1', 'dev_f1', 'min_rate', 'max_rate']
        for key in ['rate']:
            columns.remove(key)
        hyperpara_search = pd.DataFrame(data=None, columns=columns)

    # -----------Check if the parameter has been populated
    populate_check, params = populated(params, hyperpara_search)
    if populate_check:
        logger.info(f"parameter list has been trained, moving to the next...")
        return
    else:
        params['para_index'] = len(hyperpara_search)

    # -----------Train and evaluate model
    train_data, dev_data, label_encode = load_data(params)
    train_pred, dev_pred, train_shape, dev_shape, model, word_vec_map = training(train_data, dev_data, params)


    # -----------Save model
    model_path = os.path.join(*[search_path, 'save_model', str(timestamp)])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_model(label_encode, model_path + '/' + 'label_encoder_' + str(params['para_index']) + '.sav')
    save_model(word_vec_map, model_path + '/' + 'word_vec_map_' + str(params['para_index']) + '.sav')
    save_model(model, model_path + '/' + 'model_' + str(params['para_index']) + '.sav')

    # -----------Update populated table
    params['train_size'] = train_shape[0]
    params['dev_size'] = dev_shape[0]
    params['feature_num'] = train_shape[1]
    train_f1 = f1_class(train_data['encoded_label'], train_pred, params)
    params['train_f1'] = np.round(np.mean(train_f1), 4)
    params['train_f1_sg'] = list_to_str(train_f1)
    dev_f1 = f1_class(dev_data['encoded_label'], dev_pred, params)
    params['dev_f1'] = np.round(np.mean(dev_f1), 4)
    params['dev_f1_sg'] = list_to_str(dev_f1)

    target_pred_train = np.concatenate([train_data['encoded_label'].reshape(-1, 1), train_pred.reshape(-1, 1)], axis=1)
    target_pred_dev = np.concatenate([dev_data['encoded_label'].reshape(-1, 1), dev_pred.reshape(-1, 1)], axis=1)
    np.save(model_path + '/' + 'train_data' + str(params['para_index']), target_pred_train)
    np.save(model_path + '/' + 'dev_data' + str(params['para_index']),  target_pred_dev)

    single_entry = pd.Series(params)
    hyperpara_search = hyperpara_search.append(single_entry, ignore_index=True)
    hyperpara_search.drop(['para_index'], axis=1, inplace=True)
    logger.info(f"saving csv up to index:{params['para_index']}...")
    hyperpara_search.to_csv(search_path+f'/result{timestamp}.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('timestamp', type=int)
    args = parser.parse_args()

    param_filename = os.path.join(*[search_path, 'param.json'])

    with open(param_filename, 'r') as f:
        param = json.load(f)

    logger.info(f"Initializing timestamp = {args.timestamp}...")

    main(param, args.timestamp)
