import jamspell
from nltk.tokenize import casual_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import regex as re
from preprocessing import rm_punctuation


def spell_tokenizer(text):
    """
    Perform word tokenization using casual_tokenize after spelling correction
    :param text: string without punctuation
    :return: list of tokens
    """
    tokens = []
    corrector = jamspell.TSpellCorrector()
    corrector.LoadLangModel('model_en.bin')

    for word in casual_tokenize(rm_punctuation(text), preserve_case=False, reduce_len=True, strip_handles=True):
        if not (bool(re.search(r'\d', word))):
            corr_word = corrector.GetCandidates([word], 0)
            if (len(corr_word) > 0) and (word != corr_word[0]):
                for candidate in corr_word[:1]:
                    tokens.append(candidate)
            else:
                tokens.append(word)

    wordnet_lemmatizer = WordNetLemmatizer()
    stems = [wordnet_lemmatizer.lemmatize(item) for item in tokens]
    # stemmer = PorterStemmer()
    # stems = [stemmer.stem(item) for item in tokens]

    return stems


def casual_tokenizer(text):
    """
    Perform word tokenization using casual_tokenize
    :param text: string without punctuation
    :param if_cap: remove strings with numbers when appearance rate is not capped
    :return: list of tokens
    """
    tokens = [word for word in casual_tokenize(rm_punctuation(text), preserve_case=False, reduce_len=True, strip_handles=True)]

    wordnet_lemmatizer = WordNetLemmatizer()
    stems = [wordnet_lemmatizer.lemmatize(item) for item in tokens]

    # stemmer = PorterStemmer()    # Mild stemmer comparing to other type
    # stems = [stemmer.stem(item) for item in tokens]

    return stems


data_path = '/Users/sli/Projects/data'
search_path = data_path + '/mental_health_forum_data'
train_path = data_path + '/mental_health_forum_data/train_label_and_description.csv'
dev_path = data_path + '/mental_health_forum_data/dev_label_and_description.csv'

train_df = pd.read_csv(train_path, usecols=['label', 'dialog'])
dev_df = pd.read_csv(dev_path, usecols=['label', 'dialog'])


# process the datasets and save as new ones:
def main(df, type, dataset):
    if type == 'casual':
        for i in range(len(df)):
            text = df.iloc[i].dialog
            token_list = casual_tokenizer(text)
            text_reform = ' '.join(token_list)
            df = df.replace(text, text_reform)
            if i% 1000 == 0:
                print(f'Casual formating {dataset} index {i}...')
    else:
        for i in range(len(df)):
            text = df.iloc[i].dialog
            token_list = spell_tokenizer(text)
            text_reform = ' '.join(token_list)
            df = df.replace(text, text_reform)
            if i % 1000 == 0:
                print(f'Spell formating {dataset} index {i}...')

    return df


if __name__ == "__main__":

    # Process the data and save using casual tokenizer
    casual_train_data = main(train_df, type='casual', dataset='train')
    casual_dev_data = main(dev_df, type='casual', dataset='dev')

    casual_train_data.to_csv(search_path+'/casual_train_lemma.csv')
    casual_dev_data.to_csv(search_path+'/casual_dev_lemma.csv')

    # Process the data and save using spell tokenizer
    spell_train_data = main(train_df, type='spell', dataset='train')
    spell_dev_data = main(dev_df, type='spell', dataset='dev')

    spell_train_data.to_csv(search_path + '/spell_train_lemma.csv')
    spell_dev_data.to_csv(search_path + '/spell_dev_lemma.csv')