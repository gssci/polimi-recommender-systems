from __future__ import division
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from scipy import sparse
import random

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
interactions_pu = pd.read_csv('interactions.csv', delimiter="\t")
items_active = item_profiles[item_profiles['active_during_test'] == 1]

r_users = target_users['user_id'].values.tolist()

def output(matrix):
    """Matrix must be be of shape (10000,6) first column is for user_id, the remaining five are the id's of the recommended items"""
    submission = open('submission.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in matrix:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
    return

def similarity(u,v):
    """give two vectors with identical shape, returns a float"""
    C = 5
    a = np.dot(u,v)
    b = np.dot(np.linalg.norm(u),np.linalg.norm(v)) + C
    return a/b

def encode_feature(dataframe, to_encode):
    dataframe = dataframe
    #utility di sklearn che ritorna la codifica 1-of-K di un attributo
    hot = preprocessing.OneHotEncoder(dtype='int')

    for feature in to_encode:
        a = hot.fit_transform(dataframe[feature].values.reshape((-1,1))).toarray()
        for i in range(np.shape(a)[1]):
            dataframe[feature + str(i)] = a[:,i]
    return dataframe


def split_string_column(df,column,remove_outliers):
    strings = pd.unique(df[column].ravel())
    values = [str(s).split(',') for s in strings]
    flattened = [val for sublist in values for val in sublist]
    values = pd.unique(flattened)
    lenght = df[column].str.split(',', expand=True).fillna(0).shape[1]
    for col in range(lenght):
        df[column + str(col)] = df[column].str.split(',', expand=True).fillna(0).astype(int)[col]

    for index, row in df.iterrows():
        print(str(index))
        for v in values:
            if str(v) in row[column]:
                df = df.set_value(index, str(v), 1)

    for i in range(lenght):
        df = df.drop(column + str(i), 1)

    somma = np.sum(df, 0)

    if remove_outliers:
        for v in values:
            if somma[str(v)] == 1:
                df = df.drop(str(v), 1)

    return df

def interagiti(user_id):
    return np.unique(interactions_pu[interactions_pu['user_id'] == user_id]['item_id'].values)

def items_id_raccomandabili(user_id):
    raccomandabili = np.unique(items_active[~items_active.id.isin(interagiti(user_id))]['id'].values)
    return raccomandabili

def fillpop(sugg):
    a = np.unique(interactions_pu['user_id'].values)
    anonimi = [u for u in r_users if u not in a]
    uds = target_users['user_id'].values
    uds_to_index = dict(zip(uds, range(uds.size)))
    most_popular = [2778525, 1244196, 1386412, 657183, 2791339]
    smost_popular = [2778525,1244196,1386412,657183,2791339,536047]

    indexeses = [uds_to_index.get(k) for k in anonimi]

    for i in indexeses:
        sugg[i] = random.sample(smost_popular,5)
    return

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def greedy_improvement(ww,RR):
    w = list(ww)
    #max = makeEval(my_dot(w,RR))
    iteration = 0
    improving = True

    while improving:
        iteration += 1
        print('Iteration:' + str(iteration))
        improving = False

        for i in range(5,np.size(w)):
            curr_w = list(w)
            curr_w[i] += 0.05
            score = 0#makeEval(my_dot(curr_w,RR))
            if score >= max:
                max = score
                max_i = i
                improving = True

        if improving:
            w[max_i] += 0.05
            print('Current best weights: ' + str(w) + ' with MAP@5: ' + str(max))
        if not improving:
            print('Terminating with weights: ' + str(w) + ' with MAP@5: ' + str(max) + ' - number of iterations: ' + str(iteration))
    return w