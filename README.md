# fctc menuscript
Fuzzy Clustering Tree Classifier
https://ieeexplore.ieee.org/document/9684646

# manual
1. main_fctc_train_test.py is an example of train and validate in 1-fold crossvalidation
  fctc_model.fit() //train
  label_names, winner_indexes, confidences = fctc_model.predict() //validate
  output files are created in fctc_model folder consisting of
    *_result.txt //prediction results
      acc = max(h_ac_valid)
      f1 = max(h_f1_valid)
    *_rule3.txt //if-then extracted rules
    *_model_la.csv //labels of prototypes in the trained model
    *_model_za.csv //features of prototypes in the trained model

2. main_fctc_load_test.py is an example of load and test the trained model
  fctc_model.load() //load the model from fctc_model folder
  label_names, winner_indexes, confidences = fctc_model.predict() //test
