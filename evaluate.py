import torch
from torch import nn
from hyperparameters import *
from ManhattanLSTMModel import Manhattan_LSTM
from dataset import QuoraDataset
import torch.nn.utils.rnn as rnn

def my_collate(batch):
    data = [item[0] for item in batch]
    result = torch.stack([item[1] for item in batch])
    return [data, result]

use_cuda = torch.cuda.is_available()
PATH = f"output/model_epoch_{EPOCH_NUM-1}.model"

print('Reading Datafile')
validation_data = QuoraDataset(TRAIN_PATH, mode='validate')
data_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=True, collate_fn=my_collate)
embedding = validation_data.create_embedding_matrix()

model = Manhattan_LSTM(HIDDEN_SIZE, embedding)
if use_cuda:
    model = model.cuda()
model.load_state_dict(torch.load(PATH))
model.eval()

loss_fn = nn.MSELoss()


for questions, result in data_loader:
    question_1 = [question[0] for question in questions]
    question_2 = [question[1] for question in questions]
    temp = rnn.pad_sequence(question_1 + question_2)
    question_1 = temp[:, :temp.shape[1] // 2]
    question_2 = temp[:, temp.shape[1] // 2:]

    hidden = model.init_hidden(temp.shape[1] // 2)
    output_scores = model.forward([question_1, question_2], hidden).view(-1)

    print('re: ', result.item())
    print('op: ', output_scores.item())

