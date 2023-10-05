# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:46:43 2023

@author: jk
"""

import numpy as np
from .fcm import FCM

class FCTC():
  #level
  #C
  #fcm
  
  #u_class_norm
  #u_class_label
  #u_class_score
  #child
  #num_child is number of branches
  #count_live_cluster
  #x, y, jx

  def __init__(self, xMatrix=None, yVector=None, jx=None, level=0): #x is a matrix, y is a vector
    self.level = level
    self.max_level = level
    self.label_set = set(yVector) if yVector is not None else None
    C = len(self.label_set) if self.label_set is not None else 0
    self.N = xMatrix.shape[0] if xMatrix is not None else 0
    self.C = C
    self.x = xMatrix
    self.y = yVector
    if jx is None:
        jx = np.arange(len(yVector)) if yVector is not None else None
    self.jx = jx

    
    self.u_class_norm = []
    self.u_class_label = []
    self.u_class_score = []
    self.child = []

    # 1. find center of cluster
    self.fcm = FCM(self.C, self.x)

   
  def fit(self):
    self.fcm.fit()
    winner = np.argmax(self.fcm.u, axis=0)    

    # 2.&3. assign class label & score of cluster
    u_class = []
    for i in range(self.C): #cluster
      dic = dict()
      for k in self.label_set: #class
        dic[k] = 0
      u_class.append(dic)
    for j in range(self.N):
      i = winner[j] #winner cluster
      u_class[i][self.y[j]] += self.fcm.u[i][j] #u_class[cluster][class] is sum(uij)

    u_x = []
    u_y = []
    j_x = []
    for i in range(self.C): #cluster
      u_sum = sum(u_class[i].values()) #sum(u_class) each cluster
      dic = dict()
      if u_sum>0:
        for k in self.label_set: #class
          dic[k] = u_class[i][k]/u_sum #normalize u_class each class
        ucl = max(dic, key=dic.get) #winner class label
        ucs = max(dic.values()) #score of winner class
        xx = self.x[(winner==i)] #pick samples of each cluster
        yy = self.y[(winner==i)]
        jj = self.jx[(winner==i)]
      else:
        ucl = -1
        ucs = 0
        xx = []
        yy = []
        jj = []
      self.u_class_norm.append(dic)
      self.u_class_label.append(ucl)
      self.u_class_score.append(ucs)
      u_x.append(xx)
      u_y.append(yy)
      j_x.append(jj)


    # 4.1 check the same sample in multiple classes
    self.count_live_cluster = 0
    for i in range(self.C): #cluster
      if self.u_class_score[i]>0:
        self.count_live_cluster += 1

    # 4.2 deep in child
    self.child = []
    self.num_child = 0
    for i in range(self.C): #cluster
      if 0<self.u_class_score[i] and self.u_class_score[i]<1 and self.count_live_cluster>1:
        baby = FCTC(u_x[i], u_y[i], j_x[i], self.level+1)
        self.child.append(baby)
        self.num_child += 1

      else:
        self.child.append(None)

    
    for i in range(self.C):
      if self.child[i] != None:
        self.child[i].fit()
    
    for i in range(self.C):
      if self.child[i] != None:
        if self.max_level < self.child[i].max_level:
          self.max_level = self.child[i].max_level

   
  # test level 0 to height-1
  def predict(self, xMatrix1Row, height=0, fs=None): #x is matrix of 1 row
    uu, du_max = self.fcm.predict(xMatrix1Row, fs)
    winner = np.argmax(uu)
    label = self.u_class_label[winner]
    
    if self.level+1<height:
        if self.child[winner] is not None:
            label, winner = self.child[winner].predict(xMatrix1Row, height, fs)

    return label, winner

  def predicts(self, xMatrix, height=0, fs=None): #x is matrix of all rows
    labels = []
    winners = []
    for j in range(xMatrix.shape[0]):
        lb, w = self.predict(xMatrix[j:j+1], height, fs)
        labels.append(lb)
        winners.append(w)

    return labels, winners

  # save prototype za with label la to csv
  def save(self, height, fn, za=None, la=None):
    root = False
    if za is None:
      za = []
      la = []
      root = True
      
    for i in range(self.C): #cluster
      if self.child[i] is not None and self.level+1<height:
        self.child[i].save(height, fn, za, la)
      elif self.u_class_label[i]>=0:
        za.append(self.fcm.z[i])
        la.append(self.u_class_label[i])
    
    if root:
        np.savetxt(fn+"model_za.csv", za, delimiter=",")
        np.savetxt(fn+"model_la.csv", la, delimiter=",", fmt="%d")

  # save prototype za with label la to csv
  def load(self, fn):
    za = np.loadtxt(fn+"model_za.csv", delimiter=",", dtype=float)
    self.fcm.load(za) 
    self.u_class_label = self.y
    
      

  # rule extraction level 0 to height-1
  #indexed valued layer
  def rule(self, height, clus, save_rule):
    for i in range(self.C): #cluster
      clus2 = clus + "[{}]".format(i)
      msg2 = "z{} = (".format(clus2)
      msg2 = msg2 + ", ".join("{:.6f}".format(e) for e in self.fcm.z[i]) + ")"
      if self.child[i] is not None and self.level+1<height:
        save_rule.write(msg2)
        self.child[i].rule(height, clus2, save_rule)
      elif self.u_class_label[i]>=0:
         save_rule.write("{} class {}".format(msg2, self.u_class_label[i]))

  #valued flatten if-then
  def rule3(self, height, save_rule3):
    for i in range(self.C): #cluster
      if self.child[i] is not None and self.level+1<height:
        self.child[i].rule3(height, save_rule3)
      elif self.u_class_label[i]>=0:
        msg = ''
        j = 0
        for e in self.fcm.z[i]:
          if msg!='':
            msg += ' and '
          msg += "x{} is CLOSE_TO( {:.6f} )".format(j, e)
          j += 1
        save_rule3.write("if {} then y is class {}".format(msg, self.u_class_label[i]))

  #valued flatten if-then selected feature (fs is 0/1 array)
  def rule4(self, height, save_rule4, fs):  
    for i in range(self.C): #cluster
      if self.child[i] is not None and self.level+1<height:
        self.child[i].rule4(height, save_rule4, fs)
      elif self.u_class_label[i]>=0:
        msg = ''
        j = 0
        for e in self.fcm.z[i]:
          if fs[j]:
            if msg!='':
              msg += ' and '
            msg += "x{} is CLOSE_TO( {:.6f} )".format(j, e)
          j += 1
        save_rule4.write("if {} then y is class {}".format(msg, self.u_class_label[i]))

