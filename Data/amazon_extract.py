#!/usr/bin/env python
# -*- coding:utf-8 -*-

import itertools, gzip
import pandas as pd
#from utils.load_data.load_data import *
from sklearn.model_selection import train_test_split
import numpy as np



def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
    		yield eval(l)


def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
    		df[i] = d
    		i += 1
	return pd.DataFrame.from_dict(df, orient='index')


def data_preprocess(data_set, gz_path):
	
	
	data = getDF(gz_path)[['reviewerID','asin','overall','reviewText']]
	#data = getDF(gz_path)
	#print(list(data))
	data.columns = ['uid', 'iid', 'rating','learner_comment']
	# Statistics
	uids, iids = data.uid.unique(), data.iid.unique()
	n_uids, n_iids, n_ratings = len(uids), len(iids), data.shape[0]
	print('User number:', n_uids, '\tNumber of items:', n_iids, '\tNumber of ratings:', n_ratings, '\t Sparsity :', n_ratings / (n_iids * n_uids))
	print('Average number of user ratings:', n_ratings / n_uids)
	# id conversion
	uid_update = dict(zip(uids, range(n_uids)))
	iid_update = dict(zip(iids, range(n_iids)))

	data.uid = data.uid.apply(lambda x: uid_update[x])
	data.iid = data.iid.apply(lambda x: iid_update[x])
	# Data set division
	#train_idxs, test_idxs = train_test_split(list(range(n_ratings)), test_size=1)
	# Results save
	#train_data = data.iloc[train_idxs]
	#test_data = data.iloc[test_idxs]
	path_train = "data/amazon" + data_set+".csv"
	#path_test = "data/amazon" + data_set + "_test.dat"
	new_column_names = {'uid': 'learner_id','iid': 'course_id','rating': 'learner_rating','learner_comment':'learner_comment'}
	data.rename(columns=new_column_names, inplace=True)

	print(list(data))
	data.to_csv(path_train, index=False)
	#test_data.to_csv(path_test, index=False, header=None, sep='\t')
	#np.save("data/amazon" + data_set + "_id_update", [uid_update, iid_update])


if __name__ == '__main__':
	data_set = 'beauty'
	gz_path = 'data/amazon/reviews_Beauty_10.json.gz'
	# data_set = 'automotive'
	# gz_path = 'C:\\Users\\ariaschen\\Downloads\\reviews_Automotive_5.json.gz'
	# data_set = 'grocery'
	# gz_path = "../data/new_raw_data/reviews_Grocery_and_Gourmet_Food_5.json.gz"

	data_preprocess(data_set, gz_path)
		
