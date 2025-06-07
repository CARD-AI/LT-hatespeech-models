import pandas as pd
import numpy as np
import string
import json
import re

def remove_punct(text_string):
    text_string = text_string.translate(str.maketrans('', '', string.punctuation))
    text_string = text_string.replace('„', ' ')
    text_string = text_string.replace('“', ' ')
    text_string = text_string.replace('–', ' ')
    text_string = text_string.replace('   ', ' ')
    text_string = text_string.replace('  ', ' ')
    text_string = re.sub(r'(\w+)(\d{4})', r'\1 \2', text_string)
    return text_string


lrytasdf = pd.read_csv('data/lrytas_comments.csv')
lrytas_sentences_ls = list(lrytasdf.comment)

anotdf = pd.read_csv('data/anot_dataset.csv')
anot_sentences_ls = list(anotdf.data)

with open('data/wikipedia_data/lt_wiki.json', 'r') as f:
    wiki = json.load(f)

lrytas_sentences = [remove_punct(sent).lower() for sent in lrytas_sentences_ls]
anot_sentences = [remove_punct(sent).lower() for sent in anot_sentences_ls]
wiki_sentences = [sent['text'].lower() for sent in wiki]
sentences = lrytas_sentences + wiki_sentences + anot_sentences
print(len(sentences))

out_list = [{"sentence": sentence} for sentence in sentences]

with open("data/full_dataset.json", 'w', encoding='utf8') as fout:
    json.dump(out_list, fout, ensure_ascii=False)