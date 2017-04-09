from __future__ import division
import pandas as pd
import numpy as np

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')

#set da cui dopo recupero gli user_id e l'id degli items, ristretti agli items attivi e ai target users
users_r = pd.merge(user_profiles,target_users,on='user_id')
items_r = item_profiles[item_profiles['active_during_test'] == 1].reset_index().drop('index',1)

#lista degli attributi in comune tra utenti e items, ovviamente confronto users e items solo su quelli
common_attributes = np.intersect1d(item_profiles.columns.values,user_profiles.columns.values).tolist()
users = users_r[['user_id'] + common_attributes]
items = items_r[['id'] + common_attributes]

users = users.fillna(0)
items = items.fillna(0)

def p(u,i):
    return (np.count_nonzero(np.logical_and(np.logical_and(i!=np.zeros(5),u!=np.zeros(5)),i==u)) + 1) / 6

def probability(user_id,item_id):
    u = users[users['user_id']==user_id].drop('user_id',1).values
    i = items[items['id']==item_id].drop('id',1).values
    return p(u,i)

