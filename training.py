import torch
from hyperparameters import *
from ManhattanLSTMModel import Manhattan_LSTM
from dataset import QuoraDataset

use_cuda = torch.cuda.is_available()
print('Reading Datafile')
training_data = QuoraDataset(TRAIN_PATH)
data_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print('producing embedding matrix')
embedding = training_data.create_embedding_matrix(EMBEDDING_PATH)
model = Manhattan_LSTM(HIDDEN_SIZE,embedding)
if use_cuda:
    model = model.cuda()
model.init_weights()