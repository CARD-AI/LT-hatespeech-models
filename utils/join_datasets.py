import pandas as pd
import numpy as np
import os


PATH = 'data/anotated_datasets/'

onlyfiles = [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]

dataframes = []
for file in onlyfiles:
    dataframes.append(pd.read_csv(os.path.join(PATH, file)))

df = pd.concat(dataframes)
df.to_csv('data/anot_dataset_text.csv', index=False, encoding='utf8')
