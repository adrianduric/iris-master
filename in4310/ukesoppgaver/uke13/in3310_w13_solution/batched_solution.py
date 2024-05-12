import csv
import glob
import math
import random
import string
import time
import unicodedata
from io import open

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


def findFiles(path):
    return glob.glob(path)


def getnletters():
    all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
    n_letters = len(all_letters) + 1  # Plus EOS marker

    return n_letters, all_letters


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


######################################################################
# Creating the Network
# ====================
#
# This network extends `the last tutorial's RNN <#Creating-the-Network>`__
# with an extra argument for the category tensor, which is concatenated
# along with the others. The category tensor is a one-hot vector just like
# the letter input.
#
# We will interpret the output as the probability of the next letter. When
# sampling, the most likely output letter is used as the next input
# letter.
#
# I added a second linear layer ``o2o`` (after combining hidden and
# output) to give it more muscle to work with. There's also a dropout
# layer, which `randomly zeros parts of its
# input <https://arxiv.org/abs/1207.0580>`__ with a given probability
# (here 0.1) and is usually used to fuzz inputs to prevent overfitting.
# Here we're using it towards the end of the network to purposely add some
# chaos and increase sampling variety.
#
# .. figure:: https://i.imgur.com/jzVrf7f.png
#    :alt:
#
#

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, device):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.device = device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=self.nlayers)
        # Define the output layers separately outside nn.sequential so that they can be used in temperatured sampler
        self.dropout = nn.Dropout(0.1)
        self.fully_connected = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        # Output layer is just the layers above applied in sequence
        self.output_layer = nn.Sequential(
            self.dropout,
            self.fully_connected,
            self.log_softmax
        )

    def forward(self, x):
        # Format is sequence_length,batch_size,input_dimension(i.e. one-hot encoding size)
        h0 = torch.zeros(self.nlayers, x.shape[1], self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.nlayers, x.shape[1], self.hidden_dim, device=self.device)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        # Output is of shape (seq_len, batch, hidden_size):

        # Output is first flattened out along the batch and sequence dimension.
        # Since the output layer is applied independently for each time step,
        # you can treat the sequence as a batch for the output layer. Think why?
        # So you make a big batch, pass it through the output and then
        # reshape the output to get the sequence back.
        output = self.output_layer(output.view(-1, output.shape[2]))
        return output.reshape(x.shape)

    def temperatured_sample(self, temperature, all_letters, eosindex):

        h = torch.zeros(self.nlayers, 1, self.hidden_dim, device=self.device)
        c = torch.zeros(self.nlayers, 1, self.hidden_dim, device=self.device)

        startletters = list('ABCDEFGHIJKLMNOPRSTUWVY')
        letter = randomChoice(startletters)

        # output, (hn, cn)= self.lstm(x,(h0,c0))

        seq = [letter]
        z = letterToIndex(letter, all_letters)
        cur = indexToTensor(z, len(all_letters) + 1).to(self.device)

        num_chars_generated = 0
        while z != eosindex and num_chars_generated < 500:
            out, (h, c) = self.lstm(cur.view(1, 1, -1), (h, c))
            y = self.fully_connected(out[-1, :, :])
            y = y / temperature
            distr = torch.distributions.categorical.Categorical(logits=self.log_softmax(y.to(torch.device('cpu'))))
            z = distr.sample()
            # print(z)
            if z != eosindex:
                cur = indexToTensor(z, len(all_letters) + 1).to(self.device)
                seq.append(all_letters[z])

            num_chars_generated += 1

        out = ''.join(seq)
        return out

    def init_hidden(self):
        return torch.zeros(self.nlayers, 1, self.hidden_dim)


######################################################################
# Training
# =========
# Preparing for Training
# ----------------------
#
# First of all, helper functions to get random pairs of (category, line):
#


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


######################################################################
# To keep track of how long training takes I am adding a
# ``timeSince(timestamp)`` function which returns a human readable string:
#

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, all_letters):
    v = all_letters.find(letter)
    if v < 0:
        raise ValueError('letter "', letter, '" not found in all_letters')
    return v


def lineToTensor(line, n_letters, all_letters):
    sequence = [letterToIndex(i, all_letters) for i in line]
    return F.one_hot(torch.tensor(sequence), num_classes=n_letters).type(torch.float)


def lineToTarget(line, n_letters, all_letters):
    letter_indexes = [letterToIndex(line[li], all_letters) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def indexToTensor(ind, n_letters):
    tensor = torch.zeros(1, 1, n_letters)
    tensor[0][0][ind] = 1
    return tensor


def get_data():
    category_lines = {}
    all_categories = ['st']
    category_lines['st'] = []
    filterwords = ['NEXTEPISODE']

    with open('star_trek_transcripts_all_episodes_f.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el) > 1):
                    v = el.strip().replace(';', '').replace('\"',
                                                            '')  # .replace('=','') #.replace('/',' ').replace('+',' ')
                    category_lines['st'].append(v)

    print(len(all_categories), len(category_lines['st']))

    return category_lines, all_categories


def splitdata(trainratio):
    category_lines, all_categories = get_data()

    traindict = {}
    testdict = {}
    for cat in all_categories:
        np.random.shuffle(category_lines[cat])
        cut = int(len(category_lines[cat]) * trainratio)
        traindict[cat] = category_lines[cat][0:cut]
        testdict[cat] = category_lines[cat][cut:]

    return traindict, testdict, all_categories


class IterateFromDict:
    def __init__(self, adict, all_categories, batch_size=16):
        self.all_categories = all_categories
        self.dictofnames = adict
        self.inputs = []
        self.targets = []

        self.n_letters, self.all_letters = getnletters()

        print('s', self.all_categories, self.n_letters)

        for c in self.all_categories:
            for n in self.dictofnames[c]:
                self.inputs.append(lineToTensor(n, self.n_letters, self.all_letters))
                self.targets.append(lineToTarget(n, self.n_letters, self.all_letters))

        self.shuffle()
        self.counter = 0
        self.batch_size = batch_size

    def shuffle(self):
        c = list(zip(self.inputs, self.targets))
        random.shuffle(c)
        self.inputs, self.targets = zip(*c)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.inputs)

    def __next__(self):

        if self.counter >= len(self.inputs):
            # reset before raising iteration for reusal
            self.shuffle()
            self.counter = 0
            # raise
            raise StopIteration()
        else:
            start_index = self.counter
            end_index = min(self.counter + self.batch_size, len(self.inputs))
            self.counter += self.batch_size
            return pad_sequence(self.inputs[start_index: end_index]), pad_sequence(self.targets[start_index: end_index])


def run(nlayers, hidden, fname):
    gpu = False

    numepochs = 15
    hidden_dim = hidden
    batch_size = 128

    traindict, testdict, all_categories = splitdata(trainratio=0.8)

    print('Train data size', len(traindict[all_categories[0]]))

    trainit = IterateFromDict(traindict, [all_categories[0]], batch_size)
    testit = IterateFromDict(testdict, [all_categories[0]], batch_size)

    print('ntr,nte', len(trainit), len(testit))

    n_letters, all_letters = getnletters()

    if gpu:
        net = RNN(input_dim=n_letters, output_dim=n_letters, hidden_dim=hidden_dim, nlayers=nlayers,
                  device=torch.device('cuda'))
    else:
        net = RNN(input_dim=n_letters, output_dim=n_letters, hidden_dim=hidden_dim, nlayers=nlayers,
                  device=torch.device('cpu'))

    optimizer = optim.Adam(net.parameters())
    loss_function = nn.NLLLoss()

    if gpu:
        net = net.to(torch.device('cuda'))

    bestacc = 0
    for ep in range(numepochs):
        print('epoch', ep)

        avgloss = 0
        ntr = float(len(trainit))
        for i, (n, t) in enumerate(trainit):

            optimizer.zero_grad()

            if gpu:
                n = n.to(torch.device('cuda'))
            y = net(n)

            loss = 0

            if batch_size == 1:
                t = t.unsqueeze(1)
            # print(y.size(),t.size())
            for s in range(n.size()[0]):
                batchind = 0
                # print(t[s].item())
                loss += loss_function(y[s, :, :].cpu(), t[s])

            loss.backward()
            optimizer.step()

            avgloss += loss.item() / float(ntr * n.size()[0])
            if i != 0 and i % 100 == 0:
                print('epoch', ep, 'iteration', i, avgloss * ntr / float(i))

        print('Average Loss:', avgloss)

        avgloss2 = 0
        acc = 0
        nte = float(len(testit))
        for i, (n, t) in enumerate(testit):
            with torch.no_grad():
                if gpu:
                    n = n.to(torch.device('cuda'))
                y = net(n)
                _, preds = torch.max(y.data, 2)

                if batch_size == 1:
                    t = t.unsqueeze(1)
                for s in range(n.size()[0]):
                    acc += torch.sum(preds[s, :].cpu() == t[s]).item() / float(nte * n.size()[0])

                loss2 = 0
                for s in range(n.size()[0]):
                    loss2 += loss_function(y[s, :, :].cpu(), t[s]).item()
                avgloss2 += loss2 / float(nte * n.size()[0])

        print('train loss', avgloss)
        print('test loss', avgloss2)
        print('test acc', acc)
        if acc > bestacc:
            bestacc = acc
            # torch.save(net,'./model.pt')
            best_model_wts = net.state_dict()
            torch.save(best_model_wts, fname)

        temperature = 0.5
        for i in range(9):
            seq = net.temperatured_sample(temperature, all_letters, eosindex=n_letters - 1)
            print(seq)


def sampler():
    # numepochs=300
    hidden_dim = 200

    n_letters, all_letters = getnletters()
    net = RNN(input_dim=n_letters, output_dim=n_letters, hidden_dim=hidden_dim, nlayers=1)

    sdict = torch.load('./st_model_l3_h200.pt')
    net.load_state_dict(sdict)

    temperature = 0.5
    seq = net.temperatured_sample(temperature, all_letters, eosindex=n_letters - 1)
    print(seq)


if __name__ == '__main__':
    run(nlayers=3, hidden=200, fname='./st_model_l3_h200.pt')
    # sampler()


