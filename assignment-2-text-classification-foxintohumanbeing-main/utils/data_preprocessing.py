import numpy as np
import pandas as pd

def split():
    tmp = pd.read_csv('./nlp-getting-started/train.csv')
    train = int(len(tmp)*0.7)
    tmp_tr = tmp.sample(n=train, random_state=42)
    tmp_ts = tmp.loc[~tmp.index.isin(tmp_tr.index)]
    tmp_tr.reset_index().iloc[:,1:]
    tmp_ts.reset_index().iloc[:,1:]
    tmp_tr.to_csv('./nlp-getting-started/train_clean.csv')
    tmp_ts.to_csv('./nlp-getting-started/val_clean.csv')