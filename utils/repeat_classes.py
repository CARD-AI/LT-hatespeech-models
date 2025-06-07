import pandas as pd


df = pd.read_csv('data/pre-processed-dataset.csv')

labels_map = {
        'hate': 0,
        'non-hate': 1,
        'offensive': 2,
        }

df['labels'].replace(labels_map, inplace=True)
hate = df[df.labels == 0]
offensive = df[df.labels == 2]
nonh = df[df.labels == 1]

arr_non = nonh
arr_off = pd.concat([offensive] * (len(nonh) // len(offensive) + 1))
arr_hate = pd.concat([hate] * (len(nonh) // len(hate)))

arr = pd.concat([arr_non, arr_off, arr_hate])
arr = arr.sample(frac=1)
arr.to_csv('data/preprocessed_repeated.csv', index=False, encoding='utf8')
