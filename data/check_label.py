import pandas as pd

df = pd.read_csv("labels.csv")
print(df["label"].value_counts())
