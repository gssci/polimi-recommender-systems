from __future__ import division
import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", delimiter="\t")
interactions = pd.read_csv('interactions.csv', delimiter="\t")
interactions_time = interactions.sort_values(['user_id', 'created_at'], ascending=[True, False])
interactions_time = interactions_time.groupby(['user_id', 'item_id']).aggregate(np.max).reset_index().sort_values(['user_id', 'created_at'], ascending=[True, False])

aux_latestinteractions = interactions_time[['user_id', 'created_at']].groupby('user_id').first().reset_index()
latest_interactions = dict(zip(aux_latestinteractions['user_id'].values, aux_latestinteractions['created_at'].values))

current_dict = dict(zip(zip(interactions_time['user_id'].values, interactions_time['item_id'].values), interactions_time['created_at'].values))

def personalized_decay(ud, id):
    ###the result is a number in the interval [0,1] that represents
    ###how much time where 1 is the most recent and then as they get
    ###older they reduce over time
    ###1 is the most recent, and then they are progressivelly reduced.
    tau = float(60 * 60 * 24 * 7) #week in seconds
    latest = latest_interactions.get(ud, 0)
    current = current_dict.get((ud,id), 0)
    lam = 1/tau
    delta = latest - current

    return (1+(lam*delta))**(-1)

interactions_linear_time = interactions.copy()
aux = interactions_linear_time['created_at']
min_ts = min(aux[aux != 0])
max_ts = max(aux)
aux = interactions_linear_time['created_at']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
ts_scaled = min_max_scaler.fit_transform(aux)

def linear_decay(ud, id):
    return timedict.get((ud, id), 0)

interactions_linear_time['created_at'] = ts_scaled
interactions_linear_time = interactions_linear_time.drop('interaction_type', 1)
interactions_linear_time = interactions_linear_time.groupby(['user_id', 'item_id']).aggregate(np.max).reset_index()
timedict = dict(zip(zip(interactions_linear_time['user_id'].values, interactions_linear_time['item_id'].values), interactions_linear_time['created_at'].values))

interactions['time_score'] = list(map(linear_decay, interactions['user_id'].values, interactions['item_id'].values))
interactions = interactions.drop(['interaction_type', 'created_at'], 1).drop_duplicates()

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# add users/items indexes columns
interactions['user_index'] = [users_dict.get(i) for i in interactions['user_id']]
interactions['item_index'] = [items_dict.get(i) for i in interactions['item_id']]

## Scaled URM matrix ##
urm_time = sps.csr_matrix((interactions['time_score'], (interactions['user_index'], interactions['item_index'])))

