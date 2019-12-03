from re import sub
import random
import torch
from hyperparameters import *
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as split_data
import numpy as np
from gensim.models import KeyedVectors

class QuoraDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, train_ratio=0.8, pretrained_embedding_path = EMBEDDING_PATH,max_len=None, vocab_limit=None,mode='train',switch2similar=False):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.max_len = max_len
        self.vocab_size = 1
        self.vocab_limit = vocab_limit
        self.mode = mode
        self.switch2similar = switch2similar
        self.score_col = 'is_duplicate'
        self.sequence_cols = ['question1', 'question2']
        self.word2vec = KeyedVectors.load_word2vec_format(pretrained_embedding_path, binary=True)
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.vocab = set('PAD')
        self.word2index = {'PAD':0}
        self.index2word = {0:'PAD'}
        self.word2count = dict()

        self.use_cuda = torch.cuda.is_available()
        self.run()

    def __len__(self):
        if self.mode == 'train':
            return len(self.x_train)
        else:
            return len(self.x_val)

    def __getitem__(self, idx):
        #TODO sub
        if self.mode == 'train':
            return self.x_train[idx], self.y_train[idx]

        else:
            return self.x_val[idx], self.y_val[idx]

    def switch_mode(self,mode):
        self.mode = mode
    def text_to_word_list(self, text):
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

    def load_data(self):
        stops = set(stopwords.words('english'))
        #TODO should do test train split here
        data_df = pd.read_csv(self.data_file, sep='\t')

        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    # Remove unwanted words
                    if word in stops:
                        continue
                    #switch to a similar word according to word2vec
                    if self.switch2similar:
                        word = self.pick_similar_word(word)
                    if word not in self.vocab:
                        self.vocab.add(word)
                        self.word2index[word] = self.vocab_size
                        self.word2count[word] = 1
                        s2n.append(self.vocab_size)
                        self.index2word[self.vocab_size] = word
                        self.vocab_size += 1
                    else:
                        self.word2count[word] += 1
                        s2n.append(self.word2index[word])

                # Replace |sequence as word| with |sequence as number| representation
                data_df.at[index, sequence] = s2n

        return data_df
    #very expensive
    def pick_similar_word(self, word, switch_prob=0.1):
        if random.random()<switch_prob:
            similar = self.word2vec.most_similar(word)
            return similar[0][0]
        else:
            return word
    def convert_to_tensors(self):
        for data in [self.x_train, self.x_val]:
            for i, pair in enumerate(data):
                data[i][0] = torch.LongTensor(data[i][0])
                data[i][1] = torch.LongTensor(data[i][1])

                if self.use_cuda:
                    data[i][0] = data[i][0].cuda()
                    data[i][1] = data[i][1].cuda()

        self.y_train = torch.FloatTensor(self.y_train)
        self.y_val = torch.FloatTensor(self.y_val)

        if self.use_cuda:
            self.y_train = self.y_train.cuda()
            self.y_val = self.y_val.cuda()

    def run(self):
        # Loading data and building vocabulary.
        data_df = self.load_data()
        data_size = len(data_df)

        X = data_df[self.sequence_cols]
        Y = data_df[self.score_col]

        self.x_train, self.x_val, self.y_train, self.y_val = split_data(X, Y, train_size=self.train_ratio)

        # Convert labels to their numpy representations
        self.y_train = self.y_train.values
        self.y_val = self.y_val.values

        training_pairs = []
        training_scores = []
        validation_pairs = []
        validation_scores = []

        # Split to lists
        i = 0
        for index, row in self.x_train.iterrows():
            sequence_1 = row[self.sequence_cols[0]]
            sequence_2 = row[self.sequence_cols[1]]
            if len(sequence_1) > 0 and len(sequence_2) > 0:
                training_pairs.append([sequence_1, sequence_2])
                training_scores.append(float(self.y_train[i]))
            i += 1
        self.x_train = training_pairs
        self.y_train = training_scores

        print('Number of Training Positive Samples   :', sum(training_scores))
        print('Number of Training Negative Samples   :', len(training_scores) - sum(training_scores))

        i = 0
        for index, row in self.x_val.iterrows():
            sequence_1 = row[self.sequence_cols[0]]
            sequence_2 = row[self.sequence_cols[1]]
            if len(sequence_1) > 0 and len(sequence_2) > 0:
                validation_pairs.append([sequence_1, sequence_2])
                validation_scores.append(float(self.y_val[i]))
            i += 1

        self.x_val = validation_pairs
        self.y_val = validation_scores

        print('Number of Validation Positive Samples   :', sum(validation_scores))
        print('Number of Validation Negative Samples   :', len(validation_scores) - sum(validation_scores))

        assert len(self.x_train) == len(self.y_train)
        assert len(self.x_val) == len(self.y_val)

        self.convert_to_tensors()

    def create_embedding_matrix(self,pretrained_embedding_path=EMBEDDING_PATH,embedding_size=EMBEDDING_SIZE):
        self.word2vec = KeyedVectors.load_word2vec_format(pretrained_embedding_path, binary=True)
        embedding_matrix = np.zeros((len(self.word2index)+1, embedding_size))
        for word, i in self.word2index.items():
            if word not in self.word2vec.vocab:
                continue
            embedding_matrix[i] = self.word2vec[word]
        return embedding_matrix
