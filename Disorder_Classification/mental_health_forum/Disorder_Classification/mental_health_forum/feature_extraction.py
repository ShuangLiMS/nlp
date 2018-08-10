from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import casual_tokenize
from nltk.stem.porter import PorterStemmer
from preprocessing import rm_punctuation, spell_corr
import regex as re
import numpy as np
import nltk
nltk.download('punkt')

import matplotlib.pyplot as plt



class TextToVec(object):

    def __init__(self, tfidf=False, rate=(1, 1.0), binary=False, stop_words='english', analyzer='word', **kwargs):
        """
            Build vectorizer to convert list of tokens into count or tfidf vector
            :param tokenizer: specify tokenizer to use
            :param tfidf: whether to convert into count (False) or tfidf vector(True)
            :param min_rate: proportion of tokens to eliminated if appear too rare (below the rate)
            :param max_rate: proportion of tokens to eliminated if appear too often (above the rate)
            :param binary: whether to use exact count or indication of appearance
            :param stop_words: whether to eliminate stop words or not (None)
            :return:
        """
        self.tfidf = tfidf
        self.min_rate = rate[0]
        self.max_rate = rate[1]
        self.binary = binary
        self.stop_words = stop_words
        self.analyzer = analyzer
        self.vectorizer = self.build()


    def tokenizer(self, text):

        tokens = text.split(" ")

        return tokens


    def build(self):

        if self.tfidf:
            return TfidfVectorizer(
                analyzer=self.analyzer,
                tokenizer=self.tokenizer,
                stop_words=self.stop_words,
                min_df=self.min_rate,
                max_df=self.max_rate,
                binary=self.binary
            )
        else:
            return CountVectorizer(
                analyzer=self.analyzer,
                stop_words=self.stop_words,
                tokenizer=self.tokenizer,
                min_df=self.min_rate,
                max_df=self.max_rate,
                binary=self.binary
            )

    def fit(self, data):
        self.vectorizer.fit(data)


    def transform(self, data):
        return np.array(self.vectorizer.transform(data).todense())

