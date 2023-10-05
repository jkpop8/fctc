# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:28:05 2023

@author: jk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from .fctc import FCTC
from .savefile import Save

class Model:
    # fctc
    # best_h
    # mean, std
    
    
	# find best_h
	def fit(self, train_x, train_y, feat, label, fn, fold_no=0, norm=True): 
		#feat, label of validation set
	  norm_x, self.mean, self.std = normX(train_x)
	  norm_feat, mean, std = normX(feat, self.mean, self.std)
      
	  if not norm:
	     norm_x = train_x
	     norm_feat = feat
      
	  fn2 = '{}_f{}_'.format(fn, fold_no)
	  save_result = Save(fn2+'result.txt')

	  save_result.write('x_mean= '+' '.join(str(e) for e in self.mean))
	  save_result.write('x_std= '+' '.join(str(e) for e in self.std))
	  
      
	  self.fctc = FCTC(norm_x, train_y)
	  self.fctc.fit()
	  save_result.write('max_level= '+str(self.fctc.max_level)) 

	  min_diff = -1
	  max_f1 = 0
	  self.best_h = 0
	  h_f1_train = []
	  h_f1_valid = []
	  h_ac_train = []
	  h_ac_valid = []
	  start = 1 
	  for h in range(start, self.fctc.max_level+2): #height = level+1
	    ac1, f11, y_pred = self.validation(norm_x, train_y, h)
	    h_f1_train.append(f11)
	    h_ac_train.append(ac1)
        
	    ac2, f12, y_pred = self.validation(norm_feat, label, h)
	    h_f1_valid.append(f12)
	    h_ac_valid.append(ac2)
        
		
	    diff_f1 = abs(f11-f12)
	    if max_f1<f12 or (max_f1==f12 and (min_diff>diff_f1 or min_diff<0)):
	      min_diff = diff_f1
	      max_f1 = f12
	      max_ac = ac2
	      self.best_h = h

	  #jk-debug: model
	  save_result.write('h_ac_train= '+' '.join(str(e) for e in h_ac_train))
	  save_result.write('h_ac_valid= '+' '.join(str(e) for e in h_ac_valid))
	  save_result.write('h_f1_train= '+' '.join(str(e) for e in h_f1_train))
	  save_result.write('h_f1_valid= '+' '.join(str(e) for e in h_f1_valid))
	  save_result.write('max_f1= {} best_h= {}'.format(max_f1, self.best_h))
	  save_result.close()
	  plot_acc(h_ac_train, h_ac_valid, 'Accuracy', fn2+'ac')
	  plot_acc(h_f1_train, h_f1_valid, 'F1-score', fn2+'f1')
      
      #save model parameters
	  self.fctc.save(self.best_h, fn2) #za, la
	  mean_std = [self.mean, self.std]
	  np.savetxt(fn2+"model_norm.csv", mean_std, delimiter=",")	  

      #save extracted rule
	  save_rule3 = Save(fn2+'rule3.txt')
	  self.fctc.rule3(self.best_h, save_rule3)
	  save_rule3.close()
      
	  return max_ac, max_f1

	# find f1 after training
	def validation(self, featMatrix, labelVector, h=0, fs=None):
	  if h==0: h = self.best_h
	  predict, w = self.fctc.predicts(featMatrix, h, fs)

	  acc = accuracy_score(labelVector, predict)	*100
	  f1 = f1_score(labelVector, predict, average='weighted')	*100

	  #jk-debug: validation
	  # print(predict)
	  # print("confusion matrix:")
	  # cm = confusion_matrix(labelVector, predict)
	  # print(cm)
	  return acc, f1, predict
  
	def predict(self, featMatrix, h=0, fs=None, norm=True):
	  norm_feat, mean, std = normX(featMatrix, self.mean, self.std)
	  if not norm:
	     norm_feat = featMatrix
      
	  if h==0: h = self.best_h
	  predict_label, winners = self.fctc.predicts(norm_feat, h, fs)
      
	  return predict_label, winners

	def load(self, fn, fold_no=0):
	  fn2 = '{}_f{}_'.format(fn, fold_no)
	  la= np.loadtxt(fn2+"model_la.csv", delimiter=",", dtype=int)
	  self.fctc = FCTC(yVector=la)
	  self.fctc.load(fn2)
	  mean_std = np.loadtxt(fn2+"model_norm.csv", delimiter=",", dtype=float)	  
	  self.mean = mean_std[0]
	  self.std = mean_std[1]
	  self.best_h = 1
   

def plot_acc(train_acc, valid_acc, ylabel, fn):
  x_axis = [i for i in range(1,len(train_acc)+1)]
  plt.plot(x_axis, train_acc, label="Training Set", linestyle='dashed')
  plt.plot(x_axis, valid_acc, label="Validation Set")
  plt.xlabel("Height of Tree")
  plt.ylabel(ylabel+" (%)")
  plt.legend()
  plt.savefig(fn+'.png')
  plt.show()
  
def normX(x, mean=None, std=None):
    if(mean is None or std is None):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
    norm = (x-mean)/std
    norm = np.array(norm)
    return norm, mean, std