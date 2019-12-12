from re import sub
import random
import torch
import torch.utils.data
from hyperparameters import *
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split as split_data
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk.stem as stem
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('wordnet')

class QuoraDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, train_ratio=0.8, test_path=TEST_PATH, pretrained_embedding_path=EMBEDDING_PATH,
                 mode='train', normalizer=NORMALIZER):
        self.data_file = data_file
        self.test_data_file = test_path
        self.train_ratio = train_ratio
        self.vocab_size = 1
        self.mode = mode
        self.pretrained_embedding_path = pretrained_embedding_path
        self.score_col = 'is_duplicate'
        self.sequence_cols = ['question1', 'question2']
        self.word2vec = None
        self.x_train = list()
        self.y_train = list()
        self.x_val = list()
        self.y_val = list()
        self.x_test = list()
        self.test_id = list()
        self.vocab = set('PAD')
        self.word2index = {'PAD':0}
        self.index2word = {0:'PAD'}
        self.word2count = dict()
        self.normalizer = normalizer
        self.use_stop_word = USE_STOP_WORD
        self.use_cuda = torch.cuda.is_available()
        self.run()

    def __len__(self):
        if self.mode == 'train':
            return len(self.x_train)
        elif self.mode == 'validate':
            return len(self.x_val)
        else:
            return len(self.x_test)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.x_train[idx], self.y_train[idx]
        elif self.mode == 'validate':
            return self.x_val[idx], self.y_val[idx]
        else:
            return self.x_test[idx], self.test_id[idx]


    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()
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
        text = word_tokenize(text)
        normalized_sentence = []
        lancaster = stem.LancasterStemmer()
        lemmatizer = WordNetLemmatizer()

        for word in text:
            if self.normalizer == 'lancaster':
                normalized_sentence.append(lancaster.stem(word))
            elif self.normalizer == 'wordnet':
                normalized_sentence.append(lemmatizer.lemmatize(word))
            else:
                normalized_sentence.append(word)
        return normalized_sentence

    def load_data(self):
        stops = set(stopwords.words('english'))
        data_df = pd.read_csv(self.data_file, sep=',')
        # Iterate over required sequences of provided dataset
        for index, row in data_df.iterrows():
            # Iterate through the text of both questions of the row
            for sequence in self.sequence_cols:
                s2n = []  # Sequences with words replaces with indices
                for word in self.text_to_word_list(row[sequence]):
                    # Remove unwanted words
                    if self.use_stop_word and word in stops:
                        continue
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
        if self.mode == 'test':
            data_df_test = pd.read_csv(self.test_data_file, sep=',')
            for index, row in data_df_test.iterrows():
                # Iterate through the text of both questions of the row
                for sequence in self.sequence_cols:
                    s2n = []  # Sequences with words replaces with indices
                    for word in self.text_to_word_list(row[sequence]):
                        # Remove unwanted words
                        if word in stops:
                            continue
                        if word not in self.vocab:
                            self.vocab.add(word)
                            self.word2index[word] = self.vocab_size
                            self.word2count[word] = 1
                            s2n.append(self.vocab_size)
                            self.index2word[self.vocab_size] = word
                            self.vocab_size += 1
                        else:
                            s2n.append(self.word2index[word])

                    # Replace |sequence as word| with |sequence as number| representation
                    data_df_test.at[index, sequence] = s2n
            print('size of test set before {}'.format(len(data_df_test)))
            return data_df_test
        return data_df
    # very expensive

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

    def convert_test_to_tensors(self):
        for data in self.x_test:
            data[0] = torch.LongTensor(data[0])
            data[1] = torch.LongTensor(data[1])
            if self.use_cuda:
                data[0] = data[0].cuda()
                data[1] = data[1].cuda()

    def generate_test_id(self):
        self.test_id = [i for i in range(len(self.x_test))]
        self.test_id = torch.IntTensor(self.test_id)
    def run(self):
        # Loading data and building vocabulary.
        data_df = self.load_data()
        if self.mode == 'test':
            self.x_test = data_df[self.sequence_cols]
            test_pairs = []
            for index, row in self.x_test.iterrows():
                sequence_1 = row[self.sequence_cols[0]]
                sequence_2 = row[self.sequence_cols[1]]
                # if len(sequence_1) > 0 and len(sequence_2) > 0:
                if True:
                    test_pairs.append([sequence_1, sequence_2])
            self.x_test = test_pairs
            print('Number of test samples: {}'.format(len(self.x_test)))
            self.generate_test_id()
            self.convert_test_to_tensors()
        else:
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

    def create_embedding_matrix(self):
        if not self.word2vec:
            self.word2vec = KeyedVectors.load_word2vec_format(self.pretrained_embedding_path, binary=True)
        embedding_matrix = np.zeros((len(self.word2index)+1, EMBEDDING_SIZE))
        for word, i in self.word2index.items():
            if word not in self.word2vec.vocab:
                continue
            embedding_matrix[i] = self.word2vec[word]
        self.word2vec = None
        return torch.FloatTensor(embedding_matrix)
