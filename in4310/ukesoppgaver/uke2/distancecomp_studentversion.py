import os,sys,numpy as np

import torch

import time

def forloopdists(feats,protos):

  dists = np.empty((feats.shape[0], protos.shape[0]))

  for i in range(feats.shape[0]):
      for j in range(protos.shape[0]):
          dists[i, j] = np.square(np.linalg.norm(feats[i, :] - protos[j, :]))

  return dists

def numpydists(feats,protos):
  
  return feats @ protos.T
  
def pytorchdists(feats0,protos0,device):
  
  feats = torch.Tensor(feats0)
  protos = torch.Tensor(protos0)
  return feats @ protos.T

def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  feats=np.random.normal(size=(250000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))


  since = time.time()
  dists0=forloopdists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print('Comp complete in {:.3f}s'.format( time_elapsed ))


  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  print('df0',np.max(np.abs(dists1-dists0)))


  since = time.time()

  dists2=numpydists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1.numpy() - dists2)))


if __name__=='__main__':
  run()
