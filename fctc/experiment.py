# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 01:10:27 2023

@author: jk
"""

import fctc.FCTC as FCTC

def experiment(feat, label, trial=''):
  x_train = feat
  y_train = label

  #jk-debug
  # fcm = FCM(3, feat)
  # print('fcm.z=', fcm.z)
  # print('fcm.u=', fcm.u)
  # fcm.test(3, feat[0:1])
  # print('fcm.uu=', fcm.uu)

  save_result = save(fn+trial+'_result.txt')
  save_rule = save(fn+trial+'_rule.txt')
  save_rule2 = save(fn+trial+'_rule2.txt')
  save_rule3 = save(fn+trial+'_rule3.txt')
  save_rule4 = save(fn+trial+'_rule4.txt')


  if num_folds<2:
    model = Model(x_train, y_train, x_train, y_train, 0, save_result, trial)
  else:
    models, best_fold = model_kfold(num_folds, num_repeats, x_train, y_train, save_result, trial)
    model = models[best_fold]

  model[0].rule(model[1], '', save_rule)
  model[0].rule2(model[1], 'if x is', '', save_rule2)

  #flatten prototype
  za = []
  la = []
  model[0].rule3(model[1], save_rule3, za, la) #model good for implementation
  za = np.array(za)
  # print('za=', za)
  # print('la=', la)
  save_result.write('za= {}'.format(za))
  save_result.write('la= {}'.format(la))

  #feat_select
  fcm = FCM(0, x_train)
  fcm.load(za)
  uu, du_max = fcm.test(x_train)
  winner = np.argmax(uu, axis=0)
  # print('winner=', winner)
  co = [la[w] for w in winner]
  # print('co=', co)
  acc = accuracy_score(y_train, co)	*100
  print('acc all feat=', acc)
  save_result.write('acc all feat= {}'.format(acc))

  feat_rf, feat_th, feat_ff = feat_select(x_train, za, winner, fcm.dzx, 1)
  save_result.write('rf= {}'.format(feat_rf))
  save_result.write('th= {}'.format(feat_th))
  msg=''
  for f in feat_ff:
    msg = '{} {}'.format(msg, f)
  msg = 'ff {} = [{}]'.format(np.sum(feat_ff), msg)
  save_result.write(msg)
  print(np.sum(feat_ff))
  model[0].rule4(model[1], save_rule4, feat_ff) #

  #filter feature by feat_ff (** acc is down **)
  # uu, du_max = fcm.test(x_train, feat_ff)
  # winner = np.argmax(uu, axis=0)
  # # print('winner=', winner)
  # co = [la[w] for w in winner]
  # # print('co=', co)
  # acc = accuracy_score(y_train, co)	*100
  # print('acc select feat=', acc)
  # print()
  # save_result.write('acc select feat= {}\n'.format(acc))

  ac_f1 = [ [model[5], model[2]] ]
  if blind:
    ac1, f11 = Validation(model[0], blind_x, blind_y, model[1])
    save_result.write("{} {}".format(ac1, f11))
    ac_f1.append( [ac1, f11] )

  save_result.close()
  save_rule.close()
  save_rule2.close()
  save_rule3.close()
  save_rule4.close()

  #ac_f1 = [ [valid_ac, valid_f1], [blind_ac, blind_f1] ]
  return np.array([acc, feat_rf, trial, ac_f1])


	# find best_fold
	from sklearn.model_selection import KFold
	from sklearn.model_selection import StratifiedKFold
	from sklearn.model_selection import RepeatedStratifiedKFold
	def model_kfold(num_folds, num_repeats, x_train, y_train, save_result, trial=''):
	  models = []
	  best_fold = -1
	  best_f1 = 0
	  best_train_f1 = 0

	  max_ac = -1
	  min_ac = -1
	  sum_ac = 0

	  # Define the K-fold Cross Validator
	  if num_repeats>0:
		kfold = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats) 
		# all_folds = n_splites*n_repeats
	  elif num_repeats<0:
		kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
		# preserving the percentage of samples for each class
	  else: #num_repeats=0
		kfold = KFold(n_splits=num_folds, shuffle=True)
		# not preserve classes

	  fold_no = 0
	  for train, validate in kfold.split(x_train, y_train):
		save_result.write('fold_no= '+str(fold_no)) #jk-debug: kfold
		print('fold_no= ', fold_no)

		x_train_td = x_train[train, :]
		y_train_td = y_train[train]
		x_train_vd = x_train[validate, :]
		y_train_vd = y_train[validate]


		model = Model(x_train_td, y_train_td, x_train_vd, y_train_vd, fold_no, save_result, trial)
		models.append(model)
		if best_fold<0 or best_f1<model[2] or (best_f1==model[2] and best_h>model[1]) or (best_f1==model[2] and best_h==model[1] and best_train_f1<model[3][best_h-1] ) :
		  best_h = model[1]
		  best_fold = fold_no
		  best_f1 = model[2]
		  max_ac = model[5]
		  best_train_f1 = model[3][best_h-1]
		if min_ac<0 or min_ac>model[5]:
		  min_ac = model[5]
		sum_ac += model[5]
		save_result.write() #jk-debug: kfold
		fold_no += 1

	  #jk-debug: kfold
	  msg = "best_fold= {} best_f1= {} max_ac= {} best_h= {} best_train_f1= {}".format(best_fold, best_f1, max_ac, best_h, best_train_f1)
	  print(msg)
	  save_result.write(msg)
	  save_result.write("avg_ac= {} min_ac= {}".format(sum_ac/fold_no, min_ac))
	  save_result.write("f1=f1, ac=accuracy\n")
	  return models, best_fold



