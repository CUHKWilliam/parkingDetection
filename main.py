import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.signal import convolve
import wave
import time as Time
import numpy as np
import os
import math
import torch
from torch.autograd import Variable
import getData
import getDataWithoutEngine
import torch.nn.functional as F
import pickle
import getDataParallel
import sys

# torch.set_printoptions(threshold=5000)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 64000

if os.path.exists("train_data.pkl"):
    f = open("train_data.pkl", "rb")
    trainBatch = pickle.load(f)
    f.close()
    f = open("test_data.pkl", "rb")
    testBatch = pickle.load(f)
    f.close()
else:
    trainBatch, testBatch = getDataParallel.getData()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        try:
            self.input = torch.nn.Linear(len(trainBatch[0][0][0]), 200)
        except Exception:
            self.input = torch.nn.Linear(len(testBatch[0][0][0]), 200)
        self.hidden0 = torch.nn.Linear(200, 100)
        self.hidden1 = torch.nn.Linear(100, 80)
        self.hidden2 = torch.nn.Linear(80, 60)
        self.hidden3 = torch.nn.Linear(60, 40)
        self.hidden4 = torch.nn.Linear(40, 10)
        self.predict = torch.nn.Linear(10, 2)

        self.batchNorm1 = torch.nn.BatchNorm1d(100)
        self.batchNorm2 = torch.nn.BatchNorm1d(60)
        self.batchNorm3 = torch.nn.BatchNorm1d(10)

        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden0(x))
        x = self.batchNorm1(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.batchNorm2(x)
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.batchNorm3(x)
        x = self.predict(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=20, stride=1, padding=0)
        self.maxpool0 = torch.nn.MaxPool1d(kernel_size=4,stride=4,padding=0)
        self.conv1 = torch.nn.Conv1d(in_channels=4, out_channels=16, kernel_size=12, stride=1, padding=0)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=12, stride=4, padding=0)
        self.hidden0 = torch.nn.Linear(1024, 256)
        self.hidden1 = torch.nn.Linear(256, 64)
        self.hidden2 = torch.nn.Linear(64, 10)
        self.predict = torch.nn.Linear(10, 2)

        self.batchNorm0 = torch.nn.BatchNorm1d(256)
        self.batchNorm1 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.maxpool0(self.conv0(x)))
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = x.view(-1, x.size()[1]*x.size()[2])
        self.input = torch.nn.Linear(x.size()[1], 1024)
        x = F.relu(self.input(x))
        x = F.relu(self.hidden0(x))
        x = self.batchNorm0(x)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.batchNorm1(x)
        x = self.predict(x)
        return x

for i in range(len(trainBatch)):
    for j in range(len(trainBatch[i][0])):
        if len(trainBatch[i][0][j]) < len(trainBatch[0][0][0]):
            list_add0 = [0 for x in range(len(trainBatch[0][0][0]) - len(trainBatch[i][0][j]))]
            trainBatch[i][0][j] = np.append(trainBatch[i][0][j], np.array(list_add0))
        elif len(trainBatch[i][0][j]) > len(trainBatch[0][0][0]):
            trainBatch[i][0][j] = trainBatch[i][0][j][0: len(trainBatch[0][0][0])]

for i in range(len(testBatch)):
    for j in range(len(testBatch[i][0])):
        if len(testBatch[i][0][j]) < len(testBatch[0][0][0]):
            list_add0 = [0 for x in range(len(testBatch[0][0][0]) - len(testBatch[i][0][j]))]
            testBatch[i][0][j] = np.append(testBatch[i][0][j], np.array(list_add0))
        elif len(testBatch[i][0][j]) > len(testBatch[0][0][0]):
            testBatch[i][0][j] = testBatch[i][0][j][0: len(testBatch[0][0][0])]


net = Net()

loss = torch.nn.CrossEntropyLoss()

def train(flag_useTrainedModel):
    if flag_useTrainedModel:
        print("load model")
        net.load_state_dict(torch.load('model.pkl'))
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    print("train size:", len(trainBatch))
    print("batch size:", len(trainBatch[0][0]))
    for epoch in range(EPOCH):
            print('Epoch: ', epoch)
            loss_avg = 0
            correct_count = 0
            for (inputBatch, tag) in trainBatch:
                input = Variable(torch.FloatTensor(inputBatch), requires_grad=True)
                output = net(input)
                tag = torch.LongTensor(tag)
                opt.zero_grad()
                loss_output = loss(output, tag)
                loss_avg += loss_output
                loss_output.backward()
                opt.step()
                output_argmax = torch.argmax(output, dim=1)
                for i in range(len(tag)):
                    # print("this tag:", tag[i])
                    # x_data = np.arange(len(input.data[i]))
                    # y_data = np.array(input.data[i])
                    # plt.plot(x_data, y_data)
                    # plt.title("tag:"+str(tag[i].item()))
                    # plt.show()
                    if output_argmax[i].item() == tag[i].item():
                        correct_count += 1

            print("avg loss:", loss_avg/len(trainBatch))
            print("accuracy:", correct_count/float(len(trainBatch))/len(trainBatch[0][0]))
            print("save model")
            torch.save(net.state_dict(), "model.pkl")


def eval():
    net.eval()
    net.load_state_dict(torch.load('model.pkl'))
    correct_count = 0
    tagPositiveCount = 0
    tagNegativeCount = 0
    for (inputBatch, tag) in testBatch:
        input_list = []
        for i in range(len(inputBatch[0])):
            input_list.append(inputBatch[0][i])
        input_list = [input_list]
        input = Variable(torch.FloatTensor(input_list), requires_grad=True)
        output = net(input)
        tag_list = []
        for i in range(len(tag)):
            tag_list.append(tag[i])
        tag_list = torch.LongTensor(tag_list)
        output_argmax = torch.argmax(output, dim=1)

        for i in range(len(tag_list)):
            if output_argmax[i].item() == tag_list[i].item():
                correct_count += 1
            if tag_list[i].item() == 0:
                tagNegativeCount += 1
            else:
                tagPositiveCount += 1

    print("tagPositiveCount:", tagPositiveCount)
    print("tagNegativeCount:", tagNegativeCount)
    print("eval accuracy:", correct_count / float(len(testBatch)) / len(testBatch[0][0]))

def findImportant():
    net.eval()
    net.load_state_dict(torch.load('model.pkl'))
    tag_list = []

    for (inputBatch, tag) in testBatch:
        input_list = []
        for i in range(len(inputBatch[0])):
            input_list.append(inputBatch[0][i])
        input_list = [input_list]
        tag_list = []
        for i in range(len(tag)):
            tag_list.append(tag[i])
        tag_list = torch.LongTensor(tag_list)
        input = Variable(torch.FloatTensor(input_list), requires_grad=True)
        output = net(input)
        loss_output = loss(output, tag_list)
        loss_output.backward()
        grad = input.grad[0]
        # sum = 0
        # for i in range(len(grad)):
        #     sum += abs(grad[i])
        # for i in range(len(grad)):
        #     grad[i] = grad[i]/sum

        y_data = input_list[0]
        print(y_data)
        print("tag:", tag)
        x_data = [x for x in range(len(y_data))]
        plt.subplot(211)
        plt.plot(x_data, y_data)
        plt.title("freq data")
        plt.subplot(212)
        plt.plot(x_data, grad)
        plt.title("grad data")
        plt.show()

train(False)
# eval()
# findImportant()