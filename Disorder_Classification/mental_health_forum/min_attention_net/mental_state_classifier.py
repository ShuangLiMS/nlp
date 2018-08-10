"""
This model use an LSTM net to classify mental states
"""


import matplotlib
import sys
if sys.platform != "darwin":
    matplotlib.use('agg', warn=False, force =True)
import random
import pathlib
import cytoolz
import numpy
from keras.models import Sequential, model_from_json, Model
from keras.layers import LSTM, GRU, Dense, Embedding, Bidirectional, Input
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
from keras.engine import InputSpec
from keras.models import load_model
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.callbacks import TensorBoard
from attention_weight_layer import AttentionApplyLayer, AttWeightLayer
from attention_layer import AttLayer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.utils import Sequence

import en_vectors_web_lg, en_core_web_lg
import json
import os
from keras.utils.np_utils import to_categorical
import logging_util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from matplotlib import cm
from sklearn.preprocessing import LabelEncoder

logdir = "./logging"
if not os.path.exists(logdir):
    os.makedirs(logdir)
logger = logging_util.logger(__name__, logdir)


def extract_words(doc, att_words, att_sents, sents_count, max_sentence_length, nlp):

    sent_mean = np.mean(att_sents[:sents_count])
    sent_median = np.median(att_sents[:sents_count])
    sent_threshold = min(sent_mean, sent_median)

    phrases = {}

    for index, sentence in enumerate(doc.sents):
        if index >= sents_count:
            break

        if att_sents[index] < sent_threshold:
            pass
        else:
            sent_length = min(len(sentence), max_sentence_length)

            word_weights = att_words[index, :sent_length]

            #remove top 3, or 1/2 of the words
            top_num = int(min(3, 0.5 *  len(word_weights)))

            main_chunk = word_weights[np.argsort(word_weights)[:-1*top_num]]
            if len(main_chunk) > 0:
                mean_weight = np.mean(main_chunk)
                std = np.std(main_chunk)
            else:
                mean_weight = 0
                std = 0.0

            threshold = mean_weight + 3 * std # signal_to_ratio > 3

            word_indices = word_weights>threshold

            phrase_start = -100
            phrase_end = -100
            phrases_in_sent = {}
            for i, selection in enumerate(word_indices):
                if selection:
                    if i == phrase_end + 1:
                        phrase_end = i
                    else:
                        # extract the previous phrase
                        if phrase_start >= 0:
                            # "+1" to make phrase_end inclusive
                            phrase = " ".join([item.text for item in sentence[phrase_start:phrase_end+1]])

                            phrase_weight = np.max(word_weights[phrase_start:phrase_end+1])*att_sents[index]

                            if phrase not in phrases_in_sent:
                                phrases_in_sent[phrase] = phrase_weight
                            else:
                                phrases_in_sent[phrase] = max(phrases_in_sent[phrase], phrase_weight)
                        # reset for a new phrase
                        phrase_start = i
                        phrase_end = i

            # make the phrase for the last one
            if phrase_start >= 0:
                # "+1" to make phrase_end inclusive
                phrase = " ".join([item.text for item in sentence[phrase_start:phrase_end + 1]])
                phrase_weight = np.max(word_weights[phrase_start:phrase_end + 1]) * att_sents[index]
                if phrase not in phrases_in_sent:
                    phrases_in_sent[phrase] = phrase_weight
                else:
                    phrases_in_sent[phrase] = max(phrases_in_sent[phrase], phrase_weight)


            for key, value in phrases_in_sent.items():
                if key in phrases:
                    phrases[key] = max(phrases[key], value)
                else:
                    phrases[key] = value

    import operator
    sorted_phrases =  sorted(phrases.items(), key=operator.itemgetter(1), reverse=True)

    MAX_KEEP_COUNT = 2
    kept_phrase = []
    keep_count = 0
    for i, item  in enumerate(sorted_phrases):
        if item[0] not in nlp.Defaults.stop_words and not nlp(item[0])[0].is_punct:
            if keep_count < MAX_KEEP_COUNT:
                kept_phrase.append(item[0])
                keep_count+=1

    keep_count = min(len(list(doc.sents)), MAX_KEEP_COUNT)

    n_largest_ind = np.argpartition(att_sents[:len(list(doc.sents))], -keep_count)[-keep_count:]

    n_largest_ind = np.sort(n_largest_ind)

    kept_sents = [item.text for index, item in enumerate(doc.sents) if index in n_largest_ind ]

    return kept_phrase, kept_sents




def plot_weights(df, nlp, max_sentences, max_sentence_length):

    folder_name = "./runs/att_weights"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # add a new column for phrases of heavy weights
    df["phrase"] = [""]*len(df)
    df["sentences"] = [""]*len(df)

    for index, row in df.iterrows():
        try:
            text = row["dialog"]
            y_truth = row["encodedLabel"]
            y_pred = row["predicted_label"]
            y_pred_prob = row["pred_prob"][int(y_pred)]
            att_words = row["word_att"]
            att_sents = row["sent_att"]

            x_doc = nlp(text)
            x_sents = [sent.string.strip() for sent in x_doc.sents]
            sents_count = min(len(x_sents), max_sentences)
            max_length_count = np.max([min(len(nlp(sent)), max_sentence_length) for sent in x_sents])


            phrases, sentences = extract_words(x_doc, att_words, att_sents, sents_count, max_sentence_length, nlp)

            # the next two lines should be merged into one. for some reason, df.loc[index, ["phrases", "sentences"]]=[phrases, sentences] does not work
            df.at[index, "phrase"] = phrases
            df.at[index, "sentences"] = sentences

            #plt.close()
            fig = plt.figure(figsize=(10, 8))

            if sents_count > 10 or max_length_count > 50:
                # if there are very long sentences or too many sentences, draw it as a heatmap
                word_heatmap = plt.subplot(1, 2, 1)
                sent_heatmap = plt.subplot(1, 2, 2)

                word_heatmap.imshow(att_words[:sents_count, :max_length_count].astype(float), cmap="magma")
                sent_labels = []
                for i in range(sents_count):
                    sent_tokens = [item.norm_ for item in nlp(x_sents[i])]
                    sent_labels.append(sent_tokens[np.argmax(att_words[i, :len(sent_tokens)])])
                cax = sent_heatmap.imshow(np.reshape(att_sents[:sents_count].astype(float),[-1, 1]), cmap="magma")
                plt.setp(sent_heatmap, xticks=[], yticks=range(sents_count), yticklabels=sent_labels)
                plt.suptitle(f"truth: {y_truth}; pred: {y_pred}, {y_pred_prob:.4f} ")
                fig.colorbar(cax)

            else:
                #plot layouts
                sent_plots = []
                for i in range(sents_count):
                    sent_plots.append(plt.subplot(sents_count, 2, 2*i+1))
                doc_plot = plt.subplot(1, 2, 2)

                sent_labels = []
                for i in range(sents_count):
                    sent_tokens = [item.norm_ for item in nlp(x_sents[i])]
                    sent_length = np.min((len(sent_tokens), max_length_count))
                    sent_image = np.array(np.reshape(att_words[i, :sent_length], [-1, sent_length]), dtype=float)
                    sent_plots[i].imshow(sent_image, cmap="magma")
                    plt.setp(sent_plots[i], xticks=range(sent_length),
                               xticklabels=sent_tokens[:sent_length],
                             yticks=[])
                    xlabels = sent_plots[i].get_xticklabels()
                    plt.setp(xlabels, rotation=90)

                    sent_labels.append(sent_tokens[np.argmax(att_words[i, :sent_length])])

                doc_image = np.array(np.reshape(att_sents[:sents_count], [-1, 1]), dtype=float)
                cax = doc_plot.imshow(doc_image, cmap="magma")
                plt.setp(doc_plot, xticks=[], yticks = range(len(sent_labels)), yticklabels = sent_labels)

                plt.suptitle(f"truth: {y_truth}; pred: {y_pred}, {y_pred_prob:.4f}")

                fig.colorbar(cax)

            plt.tight_layout(pad=2, w_pad=1, h_pad=2)

            file_name = str(index) +"_" + '_'.join(text.split()[:5]) + ".png"
            file_name = file_name.replace("\'", "").replace("\"", "").replace("\\", "").replace("/", "")
            file_name = os.path.join(folder_name, file_name)
            fig.savefig(file_name)
            plt.close()
        except:
            logger.error(f"failed at {index}:  {row['dialog']}")

    plt.show()

    return df



class MentalStateClassifier(object):
    @classmethod
    def load(cls, path, nlp, max_sentences=20, max_sentence_length=100):
        model = load_model(path,
                       custom_objects={'AttWeightLayer': AttWeightLayer,
                                       "AttentionApplyLayer": AttentionApplyLayer})
        return cls(model, nlp, max_sentences=max_sentences, max_sentence_length=max_sentence_length)

    def __init__(self, model, nlp, max_sentences=20, max_sentence_length=100):
        self._model = model
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.nlp = nlp

    def __call__(self, doc):
        X = get_features([doc], self.nlp, max_sentences=self.max_sentences,
                         max_sentence_length=self.max_sentence_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y[0])

    def predict(self, doc):
        X = get_features([doc], self.nlp, max_sentences=self.max_sentences,
                         max_sentence_length=self.max_sentence_length)
        y = self._model.predict(X)

        return y

    # def pipe(self, docs, batch_size=1000, n_threads=2):
    #     for minibatch in cytoolz.partition_all(batch_size, docs):
    #         docs_minibatch = list(minibatch)
    #         Xs = get_features(docs_minibatch, self.max_sentences, self.max_sentence_length)
    #         ys = self._model.predict(Xs)
    #
    #         for doc, pred in zip(docs_minibatch, ys):
    #             yield {argmax(pred)}

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            docs_minibatch = list(minibatch)
            Xs = get_features(docs_minibatch, self.nlp, self.max_sentences, self.max_sentence_length)
            ys = self._model.predict(Xs)

            # create eval function for get the attention layer's output
            word_model = self._model.get_layer("time_distributed_2").layer
            word_weight_layer = self._model.get_layer("time_distributed_2").layer.get_layer("word_weights")
            word_weights = word_weight_layer.output

            sent_weight_layer = self._model.get_layer("sent_weights")
            sent_weights = sent_weight_layer.output
            word_inputs = self._model.get_layer("time_distributed_2").input

            inputs = [K.learning_phase()] + self._model.inputs
            eval_func = K.function(inputs, [sent_weights, word_inputs])
            # get the weights for all inputs, of shape [len(x), max_sample_length, 1]
            sent_model_outputs = eval_func([0] + [Xs])

            sent_att_weights = sent_model_outputs[0]
            sent_att_weights = sent_att_weights.reshape([-1, sent_att_weights.shape[1]])
            word_input_values = sent_model_outputs[1]

            word_model_inputs =[K.learning_phase()] + word_model.inputs
            eval_word_func = K.function(word_model_inputs, [word_weights])
            word_att_weights = []
            for index in range(len(minibatch)):
                doc_word_weight_values = eval_word_func([0] + [word_input_values[index, :, :]])[0]
                shape = doc_word_weight_values.shape
                doc_word_weight_values = doc_word_weight_values.reshape([shape[0], -1])
                word_att_weights.append(doc_word_weight_values)

            index = 0
            for doc, pred in zip(docs_minibatch, ys):
                index += 1
                yield {"dialog":doc.text,
                       "predicted_label": np.argmax(pred),
                       "pred_prob": pred,
                       "word_att": word_att_weights[index - 1],
                       "sent_att": sent_att_weights[index - 1, :]}


    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[1])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        # doc.user_data['my_data'] = y


def get_features(docs, nlp, max_sentences, max_sentence_length):
    """
    convert docs in sentences and words to their corresponding vocab indices
    :param docs: [["a good movie. highly recommended"], ["a not so good movie. Don't go"]]
    :param max_sentences: 100
    :param max_sentence_length: 100
    :return: [[[7, 8, 9],[99, 100]], [[8, 9, 10, 11, 12],[160, 167]]]. The numbers are not correct. This is the format.
    """
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_sentences, max_sentence_length), dtype='int32')

    for i, doc in enumerate(docs):
        if isinstance(doc, str):
            doc = nlp(doc)
        for j, sent in enumerate(doc.sents):
            if j < max_sentences:
                for k, token in enumerate(sent):
                    if k < max_sentence_length:
                        if token.norm_ in nlp.Defaults.stop_words or token.is_punct:
                            Xs[i, j, k] = 0 # to mask out the word, instead of skip the word
                        else:
                            vector_id = token.vocab.vectors.find(key=token.norm_) #token.orth)
                            if vector_id >= 0:
                                Xs[i, j, k] = vector_id
                            else:
                                Xs[i, j, k] = 0

    return Xs


class DataGenerator(Sequence):
    def __init__(self, df, nlp, batch_size, max_sentences, max_sentence_length, shuffle=True):
        self.batch_size = batch_size
        self.df = df
        self.nlp = nlp
        self.max_sentences = max_sentences
        self.max_sentence_legnth = max_sentence_length
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            self.df = self.df.sample(frac=1.0)

    def __getitem__(self, item):
        #for minibatch in cytoolz.partition_all(self.batch_size, self.df):
        #    texts = minibatch["dialog"].values
        #    labels = minibatch["encodedLabel"].values
        section = self.df[item*self.batch_size:(item+1)*self.batch_size]
        texts = section["dialog"].values
        labels = section["encodedLabel"].values
        Xs = get_features(texts, self.nlp, self.max_sentences, self.max_sentence_legnth)
        Ys = to_categorical(labels)
        return Xs, Ys


def train(train_df, dev_df,
          lstm_shape, lstm_settings, model_name, batch_size=100,
          nb_epoch=15):
    logger.info("Loading spaCy")

    nlp = en_core_web_lg.load()  # spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))


    embeddings = get_embeddings(nlp.vocab)
    if model_name == "lstm":
        model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    elif model_name == "lstm_with_attention":
        model = compile_lstm_attention(embeddings, lstm_shape, lstm_settings)
    else:  # model_name == "lstm_with_visualization":
        model = compile_visualizable_lstm_attention(embeddings, lstm_shape, lstm_settings)


    tensorboard_dir = os.path.join(logdir, "checkpoints")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tbCallBack = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True)

    logger.info("Start training...")
    model.fit_generator(
        DataGenerator(train_df, nlp=nlp, batch_size=batch_size, max_sentences=lstm_shape["max_sentences"],
                        max_sentence_length=lstm_shape["max_sentence_length"], shuffle=True),
                        validation_data=DataGenerator(dev_df, nlp=nlp,
                                                      batch_size=batch_size,max_sentences=lstm_shape["max_sentences"],
                        max_sentence_length=lstm_shape["max_sentence_length"], shuffle=False),
                        steps_per_epoch=int(np.floor(len(train_df)/batch_size)),
                        validation_steps=int(np.floor(len(dev_df)/batch_size)),
                        nb_epoch=nb_epoch, callbacks=[tbCallBack])

    return model, nlp


def compile_lstm(embeddings, shape, settings):
    logger.info("build hierachical model")
    # define the sentence encoder with attention output
    embedding_layer = Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=shape['max_sentence_length'],
        trainable=False,  # use the pre-trained word2vec. Otherwise, setting it to True to train from scratch
        weights=[embeddings],
        mask_zero=True
    )
    sentence_input = Input(shape=(shape["max_sentence_length"],), dtype='int32')
    embedding_sequences = embedding_layer(sentence_input)
    # the paper uses GRU. will come back to see if LSTM would work as well
    l_lstm = Bidirectional(LSTM(units=shape['nr_hidden'], \
                                recurrent_dropout=settings['dropout'], dropout=settings['dropout']))(
        embedding_sequences)
    sentEncoder = Model(sentence_input, l_lstm)

    # define doc level model
    review_input = Input(shape=(shape["max_sentences"], shape["max_sentence_length"],), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(units=shape['nr_hidden'],
                                     recurrent_dropout=settings['dropout'],
                                     dropout=settings['dropout']
                                     ))(review_encoder)
    preds = Dense(shape['nr_class'], activation='softmax')(l_lstm_sent)
    reviewModel = Model(review_input, preds)

    reviewModel.compile(optimizer=Adam(lr=settings['lr']), loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'])

    logger.info("model architecture: ")
    model_architecture = reviewModel.to_json()
    logger.info(json.dumps(json.loads(model_architecture), indent=2))
    logger.info("**** only the summary ****")
    logger.info(reviewModel.summary())

    return reviewModel


def compile_lstm_attention(embeddings, shape, settings):
    logger.info("build a hierarchical model with attention scheme")
    # instead of using Sequential() model, we use Model() to copy with being hierarchical

    # define the sentence encoder with attention output
    embedding_layer = Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=shape['max_sentence_length'],
        trainable=False,  # use the pre-trained word2vec. Otherwise, setting it to True to train from scratch
        weights=[embeddings],
        mask_zero=True
    )
    sentence_input = Input(shape=(shape["max_sentence_length"],), dtype='int32')
    embedding_sequences = embedding_layer(sentence_input)

    # the paper uses GRU. will come back to see if LSTM would work as well
    l_lstm = Bidirectional(GRU(shape['nr_hidden'], \
                               recurrent_dropout=settings['dropout'], dropout=settings['dropout'], \
                               return_sequences=True))(embedding_sequences)
    l_dense = TimeDistributed(Dense(shape['nr_hidden']))(l_lstm)
    l_att = AttLayer(name="word_att")(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    logger.info("**** only the summary ****")
    logger.info(sentEncoder.summary())

    # define doc level model
    review_input = Input(shape=(shape["max_sentences"], shape["max_sentence_length"]), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(shape['nr_hidden'], return_sequences=True,
                                    recurrent_dropout=settings['dropout'],
                                    dropout=settings['dropout']
                                    ))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(shape["nr_hidden"]))(l_lstm_sent)
    l_att_sent = AttLayer(name="sent_att")(l_dense_sent)
    preds = Dense(shape['nr_class'], activation='softmax')(l_att_sent)

    reviewModel = Model(review_input, preds)

    reviewModel.compile(optimizer=Adam(lr=settings['lr']), loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'])

    logger.info("model architecture: ")
    model_architecture = reviewModel.to_json()
    logger.info(json.dumps(json.loads(model_architecture), indent=2))
    logger.info("**** only the summary ****")
    logger.info(reviewModel.summary())

    return reviewModel


def compile_visualizable_lstm_attention(embeddings, shape, settings):
    logger.info("build a hierarchical model with attention scheme for visualization")
    # instead of using Sequential() model, we use Model() to copy with being hierachical

    # define the sentence encoder with attention output
    embedding_layer = Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        input_length=shape['max_sentence_length'],
        trainable=False,  # use the pre-trained word2vec. Otherwise, setting it to True to train from scratch
        weights=[embeddings],
        mask_zero=True
    )
    sentence_input = Input(shape=(shape["max_sentence_length"],), dtype='int32')
    embedding_sequences = embedding_layer(sentence_input)

    # the paper uses GRU. will come back to see if LSTM would work as well
    l_lstm = Bidirectional(GRU(shape['nr_hidden'], \
                               recurrent_dropout=settings['dropout'], dropout=settings['dropout'], \
                               return_sequences=True))(embedding_sequences)
    l_dense = TimeDistributed(Dense(shape['nr_hidden']))(l_lstm)
    l_weights = AttWeightLayer(name="word_weights")(l_dense)
    l_att = AttentionApplyLayer(name="apply_word_weights")([l_weights, l_dense])
    sentEncoder = Model(sentence_input, l_att)

    logger.info("**** only the summary ****")
    logger.info(sentEncoder.summary())

    # define doc level model
    review_input = Input(shape=(shape["max_sentences"], shape["max_sentence_length"]), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(shape['nr_hidden'], return_sequences=True,
                                    recurrent_dropout=settings['dropout'],
                                    dropout=settings['dropout']
                                    ))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(shape["nr_hidden"]))(l_lstm_sent)
    l_weights_sent = AttWeightLayer(name="sent_weights")(l_dense_sent)
    l_att_sent = AttentionApplyLayer(name="apply_sent_att")([l_weights_sent, l_dense_sent])
    preds = Dense(shape['nr_class'], activation='softmax')(l_att_sent)

    reviewModel = Model(review_input, preds)

    reviewModel.compile(optimizer=Adam(lr=settings['lr']), loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'])

    logger.info("model architecture: ")
    model_architecture = reviewModel.to_json()
    logger.info(json.dumps(json.loads(model_architecture), indent=2))
    logger.info("**** only the summary ****")
    logger.info(reviewModel.summary())

    return reviewModel


def get_embeddings(vocab):
    return vocab.vectors.data



def read_data(data_dir, data_prefix):


    file_name = os.path.join(data_dir, f"{data_prefix}_label_and_description.csv")
    df = pd.read_csv(file_name)
    df = df.sample(frac=1.0)
    return df


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", dest="dataset_dir", type=str, default="./Users/sli/Projects/data/mental_health_forum_data",
                        help="dataset dir name")

    parser.add_argument("--model_name", dest="model_name", type=str, default="lstm_with_visualization",
                        help="model's architecture [lstm, lstm_with_attention, lstm_with_visualization")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="./models",
                        help="location for the output model directory")


    parser.add_argument("--nr_hidden", dest="nr_hidden", type=int, default="64",
                        help="Number of hidden units")
    parser.add_argument("--max_sentence_length", dest="max_sentence_length", type=int, default=100,
                        help="Maximum sentence length")
    parser.add_argument("--max_sentences", dest="max_sentences", type=int, default=20,
                        help="Maximum number of sentences in a doc")

    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5,
                        help="Dropout ratio")
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001,
                        help="Learning rate")

    parser.add_argument("--nb_epoch", dest="nb_epoch", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=100,
                        help="Size of minibatches for training LSTM")

    parser.add_argument("--nr_samples", dest="nr_samples", type=int, default=-1, help="max number of samples to use")


    parser.add_argument("--inference", dest="inference", type=str, default=None, help="do eval data")

    args = parser.parse_args()

    params = vars(args)
    print("dialog parameters: ")
    print(json.dumps(params, indent=2))
    return params


def batch_inference(df, nlp, params, label_mapping):
    texts = df["dialog"].values
    encodedLabels = df["encodedLabel"].values
    labels = df["label"].values
    df_result = pd.DataFrame(columns=["dialog", "label", "predicted_label", "predicted_name", "pred_prob", "word_att", "sent_att"])

    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        if not texts[i] == doc["dialog"]:
            logger.info(f"mismatched at {i}")

        df_result = df_result.append(
            {"dialog": doc["dialog"],
             "encodedLabel": encodedLabels[i],
             "label" :labels[i],
             "predicted_label": doc["predicted_label"],
             "predicted_name": label_mapping[doc["predicted_label"]],
             "pred_prob": doc["pred_prob"],
             "word_att": doc["word_att"],
             "sent_att": doc["sent_att"]
             }, ignore_index=True)
        i += 1

    print(confusion_matrix(df_result["label"].values, np.array(df_result["predicted_name"].values, dtype=str),
                           labels=list(label_mapping.values())))

    print(classification_report(df_result["label"].values,
                                np.array(df_result["predicted_name"].values, dtype=str),
                                labels=list(label_mapping.values())))

    nlp.remove_pipe("stateClassifier")
    df = plot_weights(df_result, nlp, params["max_sentences"], params["max_sentence_length"])

    folder_name = "./runs/data_frames"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    df.to_csv(os.path.join(folder_name, "full_runtime_df.csv"))

    return df

def inference(df, model_file_name, params, label_mapping):

    def create_pipeline(nlp):
        '''
        This could be a lambda, but named functions are easier to read in Python.
        '''
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp.add_pipe(MentalStateClassifier.load(model_file_name, nlp, max_sentences=params["max_sentences"],
                                            max_sentence_length=params["max_sentence_length"]),
                     name="stateClassifier", last=True)
        logger.info(f"nlp pipeline: {nlp.pipe_names}")

    nlp = en_core_web_lg.load()

    create_pipeline(nlp)

    df = batch_inference(df, nlp, params, label_mapping)

    return df

def get_label_mapping(df):
    x_y = df.loc[:, ["label", "encodedLabel"]]
    letter_labels = x_y["label"].unique()
    mapping = {}
    for ll in letter_labels:
        targets = x_y[x_y["label"]==ll]["encodedLabel"].unique()
        if len(targets) != 1:
            logger.error(f"target is not unique for label: {ll}")
        mapping[targets[0]] = ll
    return mapping


def main():  # Training params
    params = parse_arguments()

    if params["model_dir"] is not None:
        model_dir = pathlib.Path(params["model_dir"])

    if params["inference"]:
        model_file_name = os.path.join(model_dir, params["model_name"] + ".h5")
        test_df = read_data(params["dataset_dir"], "test") #"test")
        label_mapping = get_label_mapping(test_df)

        if params["nr_samples"] != -1:
            test_df = test_df.sample(n = params["nr_samples"])

        inference(test_df, model_file_name=model_file_name, params=params, label_mapping=label_mapping)

    else:
        train_df = read_data(params["dataset_dir"], "train")
        dev_df = read_data(params["dataset_dir"], "dev")

        lstm, nlp = train(train_df, dev_df,
                     lstm_shape={ # lstm shape
                         'nr_hidden': params["nr_hidden"],
                         'max_sentences': params["max_sentences"],
                         'max_sentence_length': params["max_sentence_length"],
                         'nr_class': 7
                     },
                     lstm_settings = { # lstm_settings
                         'dropout': params["dropout"],
                         'lr': params["learning_rate"]
                     },
                     model_name=params["model_name"],
                     nb_epoch=params["nb_epoch"],
                     batch_size=params["batch_size"]
                     )

        if model_dir is not None:
            file_name = f"{params['model_name']}.h5"
            lstm.save(model_dir / file_name)
            with (model_dir / 'config.json').open("w") as file_:
                file_.write(lstm.to_json())

if __name__ == '__main__':
    #plac.call(main)
    main()
