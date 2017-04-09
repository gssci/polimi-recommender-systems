from socket import socket

import numpy as np
import random
import pandas as pd

k = 6 #length of the tail
#we pick at random the k most clicked jobs

users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
items = pd.read_csv("item_profile.csv", usecols=['id','active_during_test'], delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")
interactions2 = pd.read_csv("interactions.csv", usecols=['user_id','item_id','interaction_type'], delimiter="\t")
test_set = interactions2.groupby('user_id')['item_id'].apply(list)
#test_set = dict(interactions2.as_matrix()[:, 0:2])  # list of couples <user,items>
interactions = interactions[interactions['interaction_type'] == 1]
interactions = interactions.drop_duplicates()
interactions = interactions.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type', ascending=False).drop('user_id',1) #items clicked by the most number of users
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')

interactions[interactions.item_id.isin(items_attivi)]

def weight(i):
    return 1 - 1/i


ws = dict(zip(interactions['item_id'].values,np.apply_along_axis(weight,0,interactions['interaction_type'].values))) #dictionary of weights


##old shit
prova = pd.DataFrame({'count' : interactions.groupby(['item_id']).size()}).reset_index()
prova = prova.sort_values('count').tail(k)
cose_belle = prova.as_matrix()[:,0]

useful_items = items[items['active_during_test'] == 1]
useful_items_m = useful_items.as_matrix()

active_items = pd.read_csv('item_racc.csv', delimiter="\t")
active_items = active_items['id'].values.tolist()

new_list = [item for item in cose_belle if (item in active_items)]


def score_result(matrix):
    score = 0 #number of recommended items that were actually clicked by the user
    for row in matrix:
        user = row[0]
        for i in range(np.size(row)):
            if i == 0:
                continue
            else:
                try:
                    if row[i] in test_set[user]:
                        score = score + 1
                except KeyError:
                    continue
    return score

def fill():
    items = [0] * np.size(users)
    for j in range(10000):
        items[j] = random.sample(new_list, 5)
    return items

#result = np.column_stack((users,fill()))

