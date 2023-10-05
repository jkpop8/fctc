# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 01:06:23 2023

@author: jk
"""

import os

class Save:
  def __init__(self, fn):
    if os.path.exists(fn):
      os.remove(fn)
    self.f = open(fn, 'w')
  def write(self, msg=''):
    self.f.write(msg+"\n")
  def close(self):
    self.f.close()
