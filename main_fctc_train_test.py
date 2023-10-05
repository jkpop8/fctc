# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:36:27 2023

@author: jk
"""

import fctc.model as model
from sklearn.metrics import confusion_matrix


#prepare dataset
from sklearn.datasets import load_iris
data = load_iris()
feat = data['data'] 
label = data['target']

#prepare result file for model training
fn = "fctc_model/iris_all"
fctc_model = model.Model()
#fit(train_feature, train_label, validation_feature, validation_label, result_filename)
fctc_model.fit(feat, label, feat, label, fn) 

#example of model prediction
predict_label, winners = fctc_model.predict(feat) #predict(test_feature)
confmat = confusion_matrix(label, predict_label)
print(confmat)


