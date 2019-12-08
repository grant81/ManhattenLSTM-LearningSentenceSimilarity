import torch
from hyperparameters import *
from ManhattanLSTMModel import Manhattan_LSTM
from dataset import QuoraDataset
import torch.nn.utils.rnn as rnn
import pandas as pd


def my_collate(batch):
    data = [item[0] for item in batch]
    result = torch.stack([item[1] for item in batch])
    return [data, result]


use_cuda = torch.cuda.is_available()
print('Reading Datafile')
test_data = QuoraDataset(TRAIN_PATH, mode='test')
data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)
print('producing embedding matrix')
embedding = test_data.create_embedding_matrix()
model = Manhattan_LSTM(HIDDEN_SIZE, embedding)
model_state_dict = torch.load(PRETRAINED_PATH)
model.load_state_dict(model_state_dict)
if use_cuda:
    model = model.cuda()
model.eval()
answers = []
for questions, ids in data_loader:
    question_1 = [question[0] for question in questions]
    question_2 = [question[1] for question in questions]
    temp = rnn.pad_sequence(question_1 + question_2)
    question_1 = temp[:, :temp.shape[1] // 2]
    question_2 = temp[:, temp.shape[1] // 2:]
    hidden = model.init_hidden(temp.shape[1] // 2)
    output_scores = model.forward([question_1, question_2], hidden).view(-1)
    for idx, answer in zip(ids, output_scores):
        if answer.item() < 0.5:
            answers.append([idx.item(), 0])
        else:
            answers.append([idx.item(), 1])
result = pd.DataFrame(answers, columns=['test_id', 'is_duplicate'])
result.to_csv(TEST_OUTPUT_PATH)
