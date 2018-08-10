

import pandas as pd
import ast
import spacy
from sklearn.model_selection import train_test_split
from textacy.preprocess import preprocess_text
import tqdm

def str_clean_up(x, nlp):

    # replace the "..." by " "
    # raw example: "to...find...this...purpose...\nof...a 'voice' hearer...is...to...go.."
    out = x.replace("...", " ")

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


    # fix unicode, currency, contraction, accents
    out = preprocess_text(out,
                            fix_unicode=True,
                            transliterate=True,
                            no_currency_symbols=True,
                            no_contractions=True,
                            no_accents=True,
                            no_urls=True,
                            no_emails=True)

    # replace http and emails
    out_doc = nlp(out)
    text = []
    for token in out_doc:
        if token.like_url or token.like_email:
            pass
        else:
            text.append(token.text)
    out = " ".join(text)


    # replace other characters
    import re
    out = re.sub("[^A-Za-z0-9 ?'.:;!]+", "", out)

    return out

def clean_up(df, nlp):

    print(len(df))

    df["dialog_2"] = df["dialog"].apply(lambda x: str_clean_up(x, nlp))

    changed_df = df[df["dialog"] != df["dialog_2"]]

    for index,  row in changed_df.iterrows():
        print("----------")
        print(f"O: {row['dialog']}")
        print(f"M: {row['dialog_2']}")

    print(len(df))



    return df


def main():
    file_name = "./dataset/label_and_description.csv"
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(file_name)

    clean_up(df, nlp)



if __name__=="__main__":
    main()