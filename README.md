# fctc menuscript
Fuzzy Clustering Tree Classifier
https://ieeexplore.ieee.org/document/9684646

# manual
1. main_fctc_train_test.py is an example of train and validate in 1-fold crossvalidation<br>
  fctc_model.fit() //train<br>
  label_names, winner_indexes, confidences = fctc_model.predict() //validate<br>
  output files are created in fctc_model folder consisting of<br>
    *_result.txt //prediction results<br>
      &nbsp;&nbsp;&nbsp;acc = max(h_ac_valid)<br>
      &nbsp;&nbsp;&nbsp;f1 = max(h_f1_valid)<br>
    *_rule3.txt //if-then extracted rules<br>
    *_model_la.csv //labels of prototypes in the trained model<br>
    *_model_za.csv //features of prototypes in the trained model<br>
2. main_fctc_load_test.py is an example of load and test the trained model<br>
  fctc_model.load() //load the model from fctc_model folder<br>
  label_names, winner_indexes, confidences = fctc_model.predict() //test
