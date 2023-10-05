# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:41:01 2023

@author: jk
"""

import numpy as np

class FCM:
	# //config
	m = 2
	epsilon = 1e-4
	epoch_max = 50

	# //input/output
    # x
    # C
	# z //cluster_center[c]
	# u //membership[c][n] of train
    # dzx //

	def __init__(self, C, xMatrix=None): #x is a matrix
		self.C = C
		self.x = xMatrix

	def load(self, z):
		self.C = z.shape[0]
		self.z = z
		self.u = None

	def fit(self):
		C = self.C
		x = self.x
		N = x.shape[0]
		F = x.shape[1]

    ## initialize u
		rng = np.random.default_rng()
		u = rng.uniform(size=(C, N))
		sumu = np.sum(u, axis=1).reshape(C,1)
		self.u = u/sumu

		du_max = 0
		epoch = 0
		while True:

      ## update z
		  um =self.u ** self.m #np.power(self.u, self.m)
		  sum_ux=np.dot(um,x)
		  sum_u=np.sum(um, 1)
		  v=np.reshape(sum_u, [C,1])
		  v=np.where(v==0, 1, v)
		  self.z=sum_ux/ v

      ## update u
		  du_old = du_max
		  self.u, du_max = self.predict(x)

		  epoch +=1
		  du_chg = abs(du_max-du_old)

		  if not (du_max > self.epsilon and du_chg > self.epsilon and epoch < self.epoch_max):
		    break

	def predict(self, xMatrix, feat_ff=None): #x is a matrix
		C = self.C
		N = xMatrix.shape[0]
		F = xMatrix.shape[1]
		mp = 2.0/(self.m-1)

		Dkj, self.dzx = self.cal_dkj(xMatrix, self.z, N, C, F, feat_ff)
		u = self.cal_uij(Dkj, C, N, mp)
		du_max = np.max(abs(u-self.u)) if self.u is not None else u
		return u, du_max

	def cal_dkj(self, xMatrix, z, N, C, F, feat_ff):
	  zz = np.reshape(z, [C,1,F])
	  xx = np.repeat(xMatrix[np.newaxis, :, :], C, axis=0)
	  dd = (xx-zz)**2
    #feat_select
	  if feat_ff is not None:
	    for f in range(len(feat_ff)):
	      if feat_ff[f]==0:
	        dd[:,:,f] = 0
	  ee = np.sqrt(np.sum(dd, axis=2))
	  return ee, dd

	def cal_uij(self, dkj, C, N, mp):
	  dckj = np.repeat(dkj[np.newaxis, :, :], C, axis=0)
	  dc1j = np.reshape(dkj, [C,1,N])
	  dc1j = np.where(dc1j==0, 1, dc1j)
	  dd = (dckj/dc1j) ** mp
	  sumdd = np.sum(dd, axis=0)
	  sumdd = np.where(sumdd==0, 1, sumdd)
	  return 1.0/sumdd
