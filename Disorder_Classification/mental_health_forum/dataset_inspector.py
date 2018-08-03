import pandas as pd

df = pd.read_csv("./dataset/train_label_and_description.csv")

df["dialog_wc"] = df["dialog"].apply(lambda x: len(x.replace("\n", "").split(" ")))

df["dialog_wc"].describe()

