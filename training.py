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
print('Reading Datafile')
training_data = QuoraDataset(TRAIN_PATH)
data_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=my_collate)
print('producing embedding matrix')
embedding = training_data.create_embedding_matrix()
print('building model')
model = Manhattan_LSTM(HIDDEN_SIZE, embedding)
if use_cuda:
    model = model.cuda()
model.init_weights()
model_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = torch.optim.Adam(model_trainable_parameters, lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model.train()

total_loss = 0
total_steps = len(training_data)//BATCH_SIZE
for epoch in range(1, EPOCH_NUM + 1):
    total_loss = 0
    step = 0
    lr_scheduler.step()
    for questions, result in data_loader:
        step +=1
        question_1 = [question[0] for question in questions]
        question_2 = [question[1] for question in questions]
        temp = rnn.pad_sequence(question_1 + question_2)
        question_1 = temp[:, :temp.shape[1]//2]
        question_2 = temp[:, temp.shape[1]//2:]
        optimizer.zero_grad()
        loss = 0.0
        hidden = model.init_hidden(temp.shape[1]//2)
        output_scores = model.forward([question_1, question_2], hidden).view(-1)

        loss += loss_fn(output_scores, result)
        loss.backward()
        optimizer.step()

        total_loss += loss
        if step % 100 == 0:
            print('epoch: [{}], step: [{}/{}], curr_loss: [{:.4f}], average_loss: [{:.4f}]'
                  .format(epoch, step, total_steps, loss, total_loss / step))
    if epoch % 5 == 0:
        torch.save(model.state_dict(),
                   f"output/model_epoch_{epoch}.model")
#
# if __name__ == '__main__':
#     train_epoch()