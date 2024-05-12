from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string

import csv
import torch.distributions
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

def findFiles(path): return glob.glob(path)

def getnletters():

    all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
    n_letters = len(all_letters)+ 1 # Plus EOS marker

    return n_letters,all_letters

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


class somernn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,nlayers,device):
        super(somernn, self).__init__()
        self.hidden_dim = hidden_dim
        self.nlayers=nlayers
        self.output_dim=output_dim
        self.device=device

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim,num_layers=self.nlayers)
        self.dropout = nn.Dropout(0.1)
        self.fc=nn.Linear(self.hidden_dim,self.output_dim)
        self.ls=torch.nn.LogSoftmax(dim=1)


        #self.sampler=torch.distributions.Multinomial()

    def forward(self,x):

        #print('x', x.size())
        # format is seqlen,batch,inputdim
        h0=torch.zeros(self.nlayers,1,self.hidden_dim, device=self.device)
        c0=torch.zeros(self.nlayers,1,self.hidden_dim, device=self.device)
  
        output, (hn, cn)= self.lstm(x,(h0,c0))
        #output of shape (seq_len, batch, num_directions * hidden_size):        
        #print('o', output.size(),hn.size())
        
        y=torch.zeros((x.size()[0],x.size()[1],self.output_dim))
        for i in range(x.size()[0]):
          batchindex=0
          z=self.dropout(output[i,:,:])
          #print(z.size())
          y[i,:,:]=self.ls(self.fc(z))

        #print('y', y.size())
        #print(output[-1,:,:])
        #print(hn[:,:,:])
        return y

    def temperatured_sample(self,temperature,all_letters,eosindex):

        h=torch.zeros(self.nlayers,1,self.hidden_dim, device=self.device)
        c=torch.zeros(self.nlayers,1,self.hidden_dim, device=self.device)

        startletters=list('ABCDEFGHIJKLMNOPRSTUWVY')
        letter=randomChoice(startletters)
  
        #output, (hn, cn)= self.lstm(x,(h0,c0))

        seq=[letter]
        z=letterToIndex(letter,all_letters) 
        cur= indexToTensor(z,len(all_letters)+1).to(self.device)
        

        while z!= eosindex:
          out, (h,c) = self.lstm(cur.view(1, 1, -1), (h,c))
          y=self.fc(out[-1,:,:])
          y=y/temperature
          distr=torch.distributions.categorical.Categorical(logits=self.ls(y.to(torch.device('cpu'))   ))
          z=distr.sample()
          #print(z)
          if z!= eosindex:
            cur= indexToTensor(z,len(all_letters)+1).to(self.device)
            seq.append( all_letters[z]  )           


        out=''.join(seq)
        return out


    def initHidden(self):
        return torch.zeros(self.nlayers, 1, self.hidden_dim)


######################################################################
# Training
# =========
# Preparing for Training
# ----------------------
#
# First of all, helper functions to get random pairs of (category, line):
#

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

######################################################################
# To keep track of how long training takes I am adding a
# ``timeSince(timestamp)`` function which returns a human readable string:
#

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)




# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter,all_letters):
    v=all_letters.find(letter)
    if(v<0):
      print('letter?',letter)
      exit()
    return v


def lineToTensor(line,n_letters,all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter,all_letters)] = 1
    return tensor

def lineToTarget(line,n_letters,all_letters):
    letter_indexes = [letterToIndex(line[li],all_letters) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

def indexToTensor(ind,n_letters):
    tensor = torch.zeros(1,1, n_letters)
    tensor[0][0][ind] = 1
    return tensor

def get_data():

  category_lines = {}
  all_categories = ['st']
  category_lines['st']=[]
  filterwords=['NEXTEPISODE']

  with open('./star_trek_transcripts_all_episodes.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
      for el in row:
        if (el not in filterwords) and  (len(el)>1):
          print(el)
          v=el.strip().replace(';','').replace('\"','') #.replace('=','') #.replace('/',' ').replace('+',' ')
          category_lines['st'].append(v)

  n_categories = len(all_categories)
  print(len(all_categories), len(category_lines['st']))
  print('done')

  return category_lines,all_categories
  
  #exit()
def get_data0():
  # Build the category_lines dictionary, a list of names per language
  category_lines = {}
  all_categories = ['st']
  category_lines['st']=[]

  filterwords=['NEXTEPISODE']

  
  with open('./star_trek_transcripts_all_episodes_f.csv','w+') as f2:
    with open('./star_trek_transcripts_all_episodes.csv') as f:
      for i,line in enumerate(f):
        v=line.strip().replace('-','').replace(';','')#.replace('\"','')

        '''
        print(i)
        pos=v.find('(')
        while pos>=0:
          pos2=v.find(')')
          if(pos2<pos):

            v=v[0:pos2]+v[pos2+1:]
            
            #print('fail2',line)
            #exit()
          if pos2<0:
            #print('fail',line)
            #exit()
            break
          else:
            v=v[0:pos]+v[pos2+1:]
          pos=v.find('(')
          #print(v,pos,pos2,'\n','\n','\n','\n','\n','\n')

          if len(v)<2:
            continue

        pos=v.find('[')
        while pos>=0:
          pos2=v.find(']')
          if(pos2<pos):
            v=v[0:pos2]+v[pos2+1:]
          if pos2<0:
            print('failb',line)
            break
          v=v[0:pos]+v[pos2+1:]
          pos=v.find('[')
        f2.write(v+'\n')
        '''

        v=v.replace('=','').replace('/',' ').replace('+',' ').split(',')
        #if len(v)>1:
        #  category_lines['st'].append(v)
        for w in v:
          if (w not in filterwords) and (len(w)>1):
            category_lines['st'].append(w)


  n_categories = len(all_categories)
  print(len(all_categories), len(category_lines['st']))
  print('done')

  return category_lines,all_categories

def splitdata(trainratio):
    category_lines,all_categories=get_data()

    traindict={}
    testdict={}    
    for cat in all_categories:
        lcat=len(category_lines[cat])
        np.random.shuffle(category_lines[cat])
        cut=int(lcat*trainratio)
        traindict[cat]=category_lines[cat][0:cut]
        testdict[cat]=category_lines[cat][cut:]

    return traindict,testdict,all_categories



class iteratefromdict():
    def __init__(self, adict,all_categories):
      self.all_categories=all_categories
      self.dictofnames=adict
      self.namlab=[]

      self.n_letters,self.all_letters=getnletters()

      print('s',self.all_categories,self.n_letters)

      for i,c in enumerate(self.all_categories):
        #print('ic',i,c)
        for n in self.dictofnames[c]:
          self.namlab.append([lineToTensor(n,self.n_letters,self.all_letters),lineToTarget(n,self.n_letters,self.all_letters),i]) 
      np.random.shuffle(self.namlab)
      self.ct=0

    def num(self):
      return len(self.namlab)

    def __iter__(self):
        return self

    def __next__(self):

        if self.ct==len(self.namlab):
            # reset before raising iteration for reusal
            np.random.shuffle(self.namlab)
            self.ct=0
            #raise
            raise StopIteration() 
        else:
            self.ct+=1
            return self.namlab[self.ct-1][0],self.namlab[self.ct-1][1],self.namlab[self.ct-1][2]

    
def run(nlayers,hidden,fname):

    gpu=True

    numepochs=15
    hidden_dim=hidden

    traindict,testdict,all_categories=splitdata(trainratio=0.8)

    print('heres', traindict[all_categories[0]] )
    
    trainit=iteratefromdict(traindict,[all_categories[0]])
    testit=iteratefromdict(testdict,[all_categories[0]])
    print('heres2')
    print('ntr,nte',trainit.num(),testit.num())

    n_letters,all_letters=getnletters()

    if True==gpu:
      net=somernn(input_dim=n_letters,  output_dim=n_letters, hidden_dim=hidden_dim,   nlayers=nlayers, device=torch.device('cuda'))
    else:
      net=somernn(input_dim=n_letters,  output_dim=n_letters, hidden_dim=hidden_dim,   nlayers=nlayers, device=torch.device('cpu'))

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    loss_function = nn.NLLLoss()


    if True==gpu:
      net=net.to(torch.device('cuda'))

    bestacc=0
    for ep in range(numepochs):
      print('ep',ep)


      avgloss=0
      ntr=float(trainit.num())
      for i,(n,t,l) in enumerate(trainit):

        optimizer.zero_grad()

        if True==gpu:
          n=n.to(torch.device('cuda'))
        y=net(n)
  
        loss=0
        t=t.unsqueeze(1)
        #print(y.size(),t.size())
        for s in range(n.size()[0]):
          batchind=0
          #print(t[s].item())
          loss += loss_function(y[s,:,:].cpu(), t[s])

        loss.backward()
        optimizer.step()

        avgloss+=loss.item()/float(ntr*n.size()[0])
        if (i+1)%500==0:
          print('ep',ep,i,avgloss*ntr/float(i))


        if (i+1)%2000==0:
          temperature=0.5
          for i in range(9):
            seq=net.temperatured_sample(temperature,all_letters,eosindex=n_letters-1)
            print(seq)


      print('avgloss',avgloss)


      avgloss2=0
      acc=0
      nte=float(testit.num())
      for i,(n,t,l) in enumerate(testit):
        with torch.no_grad():
          if True==gpu:
            n=n.to(torch.device('cuda'))
          y=net(n)
          _, preds = torch.max(y.data, 2)

          #print('preds',preds.size())

          t=t.unsqueeze(1)
          for s in range(n.size()[0]):
            batchindex=0
            acc+=torch.sum(preds[s,:].cpu() == t[s] ).item()/float(nte*n.size()[0])

          loss2=0
          for s in range(n.size()[0]):
            batchind=0
            loss2 += loss_function(y[s,:,:].cpu(), t[s]).item()
          avgloss2+=loss2/float(nte*n.size()[0])

          if (i+1)%1000==0:
            print('test ep',ep,i)
          #  print(acc/float(i))

      #shouldbe class-wise :), not banana average
      print('trainloss',avgloss)
      print('test loss', avgloss2 )
      print('test acc', acc )
      if acc> bestacc:
        bestacc=acc
        #torch.save(net,'./model.pt')
        best_model_wts = net.state_dict()
        torch.save(best_model_wts,fname)

      temperature=0.5
      for i in range(9):
        seq=net.temperatured_sample(temperature,all_letters,eosindex=n_letters-1)
        print(seq)



def sampler():

  #numepochs=300
  hidden_dim=200


  n_letters,all_letters=getnletters()
  net=somernn(input_dim=n_letters,  output_dim=n_letters, hidden_dim=hidden_dim,   nlayers=1)

  sdict=torch.load('./st_model_l3_h200.pt')
  net.load_state_dict(sdict)


  temperature=0.5
  seq=net.temperatured_sample(temperature,all_letters,eosindex=n_letters-1)
  print(seq)


if __name__=='__main__':
    run(nlayers=3,hidden=200,fname='./st_model_l3_h200.pt')
    #sampler()
  

