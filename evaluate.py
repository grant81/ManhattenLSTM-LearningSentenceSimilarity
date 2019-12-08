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


true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
total_loss = 0

accuracy = 0.0
precision = 0.0
recall = 0.0

scale = 1.0
use_cuda = torch.cuda.is_available()
print('Reading Datafile')
validate_data = QuoraDataset(TRAIN_PATH, mode='validate')
data_loader = torch.utils.data.DataLoader(validate_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=my_collate)
print('producing embedding matrix')
embedding = validate_data.create_embedding_matrix()
print('building model')
model = Manhattan_LSTM(HIDDEN_SIZE, embedding)
model_state_dict = torch.load(PRETRAINED_PATH)
model.load_state_dict(model_state_dict)
if use_cuda:
    model = model.cuda()
model.eval()
loss_fn = nn.MSELoss()
for questions, results in data_loader:

    question_1 = [question[0] for question in questions]
    question_2 = [question[1] for question in questions]
    temp = rnn.pad_sequence(question_1 + question_2)
    question_1 = temp[:, :temp.shape[1] // 2]
    question_2 = temp[:, temp.shape[1] // 2:]
    hidden = model.init_hidden(temp.shape[1] // 2)
    output_scores = model.forward([question_1, question_2], hidden).view(-1)
    total_loss += loss_fn(output_scores, results)

    for actual, predict in zip(results, output_scores):
        if actual.item() < 0.5 and predict.item() < 0.5:
            true_negative += 1

        if actual.item() < 0.5 and predict.item() >= 0.5:
            false_positive += 1

        elif actual.item() >= 0.5 and predict.item() >= 0.5:
            true_positive += 1

        if actual.item() >= 0.5 and predict.item() < 0.5:
            false_negative += 1

accuracy = (true_positive + true_negative) * 100 / len(validate_data)
if true_positive + false_positive > 0:
    precision = true_positive * 100 / (true_positive + false_positive)
if true_positive + false_negative > 0:
    recall = true_positive * 100 / (true_positive + false_negative)

print('Validation Accuracy: %f Validation Precision: %f Validation Recall: %f Validation Loss: %f' % (
    accuracy, precision, recall, total_loss))
