from nltk.corpus import wordnet
import nltk
import random

nltk.download('wordnet')
from re import sub
from hyperparameters import *
import pandas as pd


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = sub(r"what's", "what is ", text)
    text = sub(r"\'s", " ", text)
    text = sub(r"\'ve", " have ", text)
    text = sub(r"can't", "cannot ", text)
    text = sub(r"n't", " not ", text)
    text = sub(r"i'm", "i am ", text)
    text = sub(r"\'re", " are ", text)
    text = sub(r"\'d", " would ", text)
    text = sub(r"\'ll", " will ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\/", " ", text)
    text = sub(r"\^", " ^ ", text)
    text = sub(r"\+", " + ", text)
    text = sub(r"\-", " - ", text)
    text = sub(r"\=", " = ", text)
    text = sub(r"'", " ", text)
    text = sub(r"(\d+)(k)", r"\g<1>000", text)
    text = sub(r":", " : ", text)
    text = sub(r" e g ", " eg ", text)
    text = sub(r" b g ", " bg ", text)
    text = sub(r" u s ", " american ", text)
    text = sub(r"\0s", "0", text)
    text = sub(r" 9 11 ", "911", text)
    text = sub(r"e - mail", "email", text)
    text = sub(r"j k", "jk", text)
    text = sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def pick_similar_word(word, switch_prob=0.5):
    if random.random() < switch_prob:
        similar = wordnet.synsets(word)
        if len(similar) > 0:
            word = similar[random.randint(0, len(similar) - 1)].lemmas()[0].name()
        return word
    else:
        return word


data_df = pd.read_csv(TRAIN_PATH, sep=',')
last_row = data_df.tail(1)
id = last_row['id'].values[0] + 1
qid = last_row['qid2'].values[0] + 1
additional = []
score_col = 'is_duplicate'
sequence_cols = ['question1', 'question2']
for index, row in data_df.iterrows():
    cur = [id, qid, qid + 1]
    for seq in sequence_cols:
        question = ''
        for word in text_to_word_list(row[seq]):
            question += pick_similar_word(word)
            question += ' '
        cur.append(question)
    cur.append(row[score_col])
    additional.append(cur)
    id += 1
    qid += 2
additional = pd.DataFrame(additional, columns=["id", "qid1", "qid2", "question1", "question2", "is_duplicate"])
out = pd.concat([data_df, additional], ignore_index=True)
out.to_csv(ADDITIONAL_TRAINING_PATH, index=False)
