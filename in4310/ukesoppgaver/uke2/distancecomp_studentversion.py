import os,sys,numpy as np

import torch

import time

def forloopdists(feats,protos):

  dists = np.empty((feats.shape[0], protos.shape[0]))

  for i in range(feats.shape[0]):
      for j in range(protos.shape[0]):
          diff = feats[i, :] - protos[j, :]
          dists[i, j] = np.dot(diff, diff)

  return dists

def numpydists(feats,protos):
  
  feats_expanded = feats.reshape(-1, 1, 300)
  protos_expanded = protos.reshape(1, -1, 300)
  # feats_expanded = feats[:, np.newaxis, :]
  # protos_expanded = protos[np.newaxis, :, :]

  diff = feats_expanded - protos_expanded
  # sq_norm = np.square(np.linalg.norm(diff, axis=2))
  sq_norm = np.einsum("ijk, ijk -> ij", diff, diff)

  return sq_norm

def pytorchdists(feats0,protos0,device):
  
  feats = torch.tensor(feats0)
  protos = torch.tensor(protos0)

  feats_expanded = feats.reshape(-1, 1, 300)
  protos_expanded = protos.reshape(1, -1, 300)

  diff = feats_expanded - protos_expanded
  # sq_norm = torch.square(torch.linalg.vector_norm(diff, dim=2))
  sq_norm = torch.einsum("ijk, ijk -> ij", diff, diff)

  return sq_norm

def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  feats=np.random.normal(size=(5000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))


  since = time.time()
  dists0=forloopdists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists0.shape)


  device=torch.device('cpu')
  since = time.time()

  dists1=pytorchdists(feats,protos,device)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))
  print(dists1.shape)

  print('df0',np.max(np.abs(dists1.numpy() - dists0)))


  since = time.time()

  dists2=numpydists(feats,protos)


  time_elapsed=float(time.time()) - float(since)

  print('Comp complete in {:.3f}s'.format( time_elapsed ))

  print(dists2.shape)

  print('df',np.max(np.abs(dists1.numpy() - dists2)))

  print("np vs loop", np.max(np.abs(dists0 - dists2)))

if __name__=='__main__':
  run()
