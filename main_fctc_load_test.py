# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:36:27 2023

@author: jk
"""

import fctc.model as model
from sklearn.metrics import confusion_matrix
import numpy as np


#prepare dataset
from sklearn.datasets import load_iris
data = load_iris()
feat = data['data'] 
label = data['target']

#prepare model file from training
fn = "fctc_model/iris_all"
fctc_model = model.Model()
fctc_model.load(fn) #load(model_filename, fold_no=0)

#example of model prediction
predict_label, winners, confis = fctc_model.predict(feat) #predict(test_feature, norm=True)

# Convert NumPy types to standard Python types for confis
confidences = [
    dict((int(k), float(v)) for k, v in d)
    for d in confis
]
a = np.array([label, predict_label, winners, confidences], dtype=object)
print(a.transpose())

confmat = confusion_matrix(label, predict_label)
print(confmat)
