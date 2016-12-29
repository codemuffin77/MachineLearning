###This program is designed to automated recommend movies for users
###It mimic the users' behavior to score any given movie
###recommend movies based on the similarity or high quality (top rating score)
###This program is free to use, you can redistribute it
###and modify it under the software license.
###Program Author: Zhen Qian (Martin), Rutgers University
###Email:qianzhen77@hotmail.com
###This program is distributed in the hope that it will be useful,
###but without any warranty for a particular purpose.
###This program is passed by python 2.7.
### Copyright (c) Nov 2016

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 10:58:30 2016

@author: martin
"""

import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.spatial.distance import pdist, squareform

###formats function is used to tranfer the import excel format to 2D matrix
###excel format: UserID	MovieID	rating	timstamp
#######           196	     242	      3	      881250949
######            186	     302	      3	      891717742
###### 2D matrix: column attribute: userID     row attribute: movieID 
######input: 
######     df (excel format)
######output:
######     D (2D matrix)
######     m (D's column)
######     n (D's row)

def formats(df):
  arr=df.as_matrix(columns=df.columns[:])
  rows, row_pos = np.unique(arr[:, 0], return_inverse=True)
  cols, col_pos = np.unique(arr[:, 1], return_inverse=True)
  rows, row_pos = np.unique(arr[:, 0], return_inverse=True)
  cols, col_pos = np.unique(arr[:, 1], return_inverse=True)
  D = np.zeros((len(rows), len(cols)), dtype=arr.dtype)
  D [row_pos, col_pos] = arr[:, 2]
  D = ma.masked_array(D, mask=D==0)
  m, n = D.shape
  return D, m, n

###mask function is labeled the NA data sets, so the total errors will only calculate 
###the distances between avaiable elements in old matrix and corresponding elements 
###in reconstructed new matrix
###input:
###     D: orignal matrix [m*n]
###output:
###     D: NA data points mask with zeros
def mask(D):
  return ma.getmaskarray(D) if ma.isMA(D) else np.zeros(D.shape, dtype=bool)

###matrixFactorization function is used to calculate (1) total error (2) MSE  
###input: 
###     D: original matrix [m*n] 
###     P: initial matrix with random numbers [m*K]
###     Q: initial matrix with random numbers [K*n]
###output:
###     mse: square root of distances
###     P: optimised matrix [m*K]
###     Q: optimised matrix [K*n]

def matrixFactorization(D, P, Q, steps=3000, alpha=0.0002, beta=0.02):
  K = P.shape[1]
  P = np.copy(P)
  Q = np.copy(Q)

  for step in xrange(steps):
    for i in xrange(len(D)):
      for j in xrange(len(D[i])):
        if not mask(D)[i, j]:
          eij = D[i, j] - np.dot(P[i, :], Q[:, j])
          
          for k in xrange(K):
            P[i, k] = P[i, k] + alpha * (2 * eij * Q[k, j] - beta * P[i, k])
            Q[k, j] = Q[k, j] + alpha * (2 * eij * P[i, k] - beta * Q[k, j])
            
    mse= np.sqrt((D-np.dot(P,Q))**2).mean()
    
    print 'error:' , eij, 'mse:', mse, 'step:', step
    
  return P, Q

if __name__ == '__main__':
    
###Part I: import original data sets
###small scale mock test: data10.xls, run this program to make sure codes are OK
###train file:data10k.xls , run this program to make sure hardware is ok.
###Cross-validation file: data10kCV.xls
###more data files: data20k.xls, data30k.xls, data40k.xls, data50k.xls, data65k.xls
    
  df = pd.read_excel('data10.xls', sep='\s+',names=['UserId','MovieId','rating','timestamp'])
  #df = pd.read_excel('data10k.xls', sep='\s+',names=['UserId','MovieId','rating','timestamp']) 
  #df = pd.read_excel('data30k.xls', sep='\s+',names=['UserId','MovieId','rating','timestamp'])    
  #df = pd.read_excel('data10kCV.xls', sep='\s+',names=['UserId','MovieId','rating','timestamp'])    
  #clos: movie ID; rows: UserID    

###Part II: tranfer data format and get 2D matrix
###get the 2D matrix, including userID and movieID
  D, m, n = formats(df)
### K is tradeoff between bias and variance
### K is tested from 1 to 20, and K=10 is optimised selection
  K = 10
  
  
###part III: generate reconstructed matrix (new matrix)
### initial parameters
  P = np.random.rand(m, K)
  Q = np.random.rand(K, n)
  
###print function  
  np.set_printoptions(formatter={'all': lambda x: str(x).rjust(2)})
  np.set_printoptions(precision = 2, formatter=None)
  
###use matrix factorization method to get matrix decomposition P, Q
###D (old matrix): [m*n]
###P (matrix decomposition part): [m*K]
###Q (matrix decomposition part): [K*n]
###nD (new matrix, reconstructed matrix): [m*n]
  P, Q = matrixFactorization(D, P, Q)
  nD = np.dot(P, Q)
  print 'D matrix\n', D, '\n'
  print 'P matrix decomposition\n', P, '\n'
  print 'Q matrix decomposition\n', Q, '\n'
  print 'reconstructed D matrix\n', np.dot(P, Q), '\n'
  
###output the nD matrix into the txt file and easy to import it for saving time  
#  np.savetxt('nR-20k.txt', nD, fmt='%-7.5f')

###part IV: recommend movie based on similarity  
  
  distances=np.sqrt((D-nD)**2)
  pairDist=squareform(pdist(np.transpose(distances)))
  upper = np.triu(pairDist, 1)
  
  #assume movieID=8
#  idx = np.where((upper == upper[np.nonzero(upper)].min()))
#  search=col[7]  
#  relation=[]
#  relation1 = upper [0:7, 7]
#  relation2 = upper [7,8:1253]
#  value = min (relation2)
#  print np.where( relation2== relation2.min())
#  print cols[137]
  #result array([130], mean 130+7=137 is originl index, so movieID=138
#  idx = np.where(value)
#  relation.append(upper[7,:])
  
  #assume movieID =30
#  sear=cols[29]
#  rel=[]
#  rel1 = upper [0:29, 29]
#  rel2 = upper [29,30:1253]
#  value1= min (rel1)
#  value2 = min (rel2)
#  print np.where( rel2== rel2.min())
#  print cols[1215]

###part V: recommend movies for high quality  

  scoreM= np.matrix(nD)
  score= scoreM.mean(0)
  b=np.array(score)
###select the top 3 highest rating score movies
  indexs = (-b).argsort()[:3]
  c=np.unique(b)
  c_r=c[::-1]
###print the movies info
#  print b[0,1408], cols[1408]
#  print b[0,1468], cols[1468]
#  print b[0,1158], cols[1158]
  
