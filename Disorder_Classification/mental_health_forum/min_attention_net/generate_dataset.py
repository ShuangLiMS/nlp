import pandas as pd
import ast
from textacy.preprocess import preprocess_text

from sklearn.model_selection import train_test_split

import spacy

def load_file(file_name):
    df_orig = pd.read_csv(file_name)

    # clean up NaN rows
    df_orig = df_orig.dropna(subset=["dialog"])

    # convert dialog field back to its dialog form
    df_orig.loc[:, "dialog"] = df_orig["dialog"].apply(lambda x: ast.literal_eval(x))

    df_orig = df_orig[df_orig["dialog"].apply(lambda x: len(x) > 0)]

    df = df_orig.loc[:, ["label", "title", "dialog"]]


    # compute the len for later use when statistics are computed
    df.loc[:, "dialog_len"] = df["dialog"].apply(lambda x: len(x))

    # clean up the advertisement and other non dialogs
    df["valid"] = df["dialog"].apply(lambda x: not x[0].startswith("========="))
    df = df[df["valid"] == True]

    return df


def str_clean_up(x, nlp):


    # fix unicode, currency, contraction, accents
    text = preprocess_text(x,
                          fix_unicode=True,
                          transliterate=True,
                          no_currency_symbols=True,
                          no_contractions=True,
                          no_accents=True)


    # replace http and emails
    text_doc = nlp(text)
    text_list = []
    for token in text_doc:
        if token.like_url or token.like_email:
            pass
        else:
            text_list.append(token.text)

    out = " ".join(text_list)

    out = preprocess_text(out, no_urls = True, no_emails = True)

    # replace the "..." by " "
    # raw example: "to...find...this...purpose...\nof...a 'voice' hearer...is...to...go.."
    out = out.replace("...", " ")

    out = out.replace("\n", " ")

    # replace all punctuations
    # out = out.replace('.', '')
    # out = out.replace(',', '')
    # out = out.replace('"', '')
    # out = out.replace('?', '')
    # out = out.replace('!', '')

    # replace all slashes
    out = out.replace('\\', '')
    out = out.replace('/', '')


    # replace other characters
    import re
    out = re.sub("[^A-Za-z0-9 ?'.:;!]+", "", out)

    return out


def main():
    file_names = {
        "anxiety": "/Users/sli/Projects/data/mental_health_forum_data/anxiety_threads_w_dialogs.csv",
        "bipolar": "/Users/sli/Projects/data/mental_health_forum_data/Bipolar_threads_w_dialogs.csv",
        "depression": "/Users/sli/Projects/data/mental_health_forum_data/depression_threads_w_dialogs.csv",
        "hearing_voices": "/Users/sli/Projects/data/mental_health_forum_data/Hearing Voices_threads_w_dialogs.csv",
        "PTSD": "/Users/sli/Projects/data/mental_health_forum_data/PTSD_threads_w_dialogs.csv",
        "Schizophrenia": "/Users/sli/Projects/data/mental_health_forum_data/Schizophrenia_threads_w_dialogs.csv",
        "self_harm": "/Users/sli/Projects/data/mental_health_forum_data/self_harm_threads_w_dialogs.csv"
    }

    dataset = None
    for k, v in file_names.items():
        df_k = load_file(v)
        print(f"{k}: {df_k['dialog_len'].describe()}")
        if k in ["bipolar", "depression"]:
            chosen_df = df_k.sample(6000)
        else:
            chosen_df = df_k

        # change the labels
        chosen_df.loc[:, "label"] = k

        # use only the first entry as
        chosen_df["dialog"] = chosen_df["dialog"].apply(lambda x: x[0].strip())
        chosen_df["dialog_len"] = chosen_df["dialog"].apply(lambda x: len(x.split(" ")))

        # drop the rows with nan entry in the dialog field
        chosen_df = chosen_df.dropna(subset=["dialog"])

        # drop the rows with empty string in the dialog field
        chosen_df = chosen_df[chosen_df["dialog_len"]>0]

        dataset = pd.concat([dataset, chosen_df], ignore_index=True)

    dataset = dataset[dataset["dialog_len"]>4]
    dataset = dataset.dropna(subset=["dialog"])

    keys = list(file_names.keys())
    dataset["encodedLabel"] = dataset["label"].apply(lambda x: keys.index(x))

    # do some clean up
    nlp = spacy.load("en_core_web_sm")
    dataset["dialog"] = dataset["dialog"].apply(lambda x: str_clean_up(x, nlp))

    #shuffle the data
    dataset = dataset.sample(frac=1.0)

    #split it into train, dev, test portions
    train_dev, test_df = train_test_split(dataset, test_size=0.15)
    train_df, dev_df = train_test_split(train_dev, test_size=.20)

    #save them
    print("saving data dataset...")
    dataset.to_csv("/Users/sli/Projects/data/mental_health_forum_data/label_and_description.csv")
    print("saving data train_df...")
    train_df.to_csv("/Users/sli/Projects/data/mental_health_forum_data/train_label_and_description.csv")
    print("saving data dev_df...")
    dev_df.to_csv("/Users/sli/Projects/data/mental_health_forum_data/dev_label_and_description.csv")
    print("saving data test_df...")
    test_df.to_csv("/Users/sli/Projects/data/mental_health_forum_data/test_label_and_description.csv")


if __name__=="__main__":
    main()