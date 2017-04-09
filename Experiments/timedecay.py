from __future__ import division
import numpy as np
import pandas as pd
from scipy import sparse as sps
from sklearn import preprocessing
import sklearn.metrics.pairwise as metrics

users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", delimiter="\t")
td = pd.read_csv('interactions.csv', delimiter="\t")
times = td.sort_values(['user_id', 'created_at'], ascending=[True, False])
times = times.groupby(['user_id','item_id']).aggregate(np.max).reset_index().sort_values(['user_id','created_at'],ascending=[True,False])

latest_grp = times[['user_id','created_at']].groupby('user_id').first().reset_index()
latest_dict = dict(zip(latest_grp['user_id'].values,latest_grp['created_at'].values))

current_dict = dict(zip(zip(times['user_id'].values,times['item_id'].values),times['created_at'].values))

def time_decay(ud,id):
    tau = float(60 * 60 * 24 * 7) #week in seconds
    latest = latest_dict.get(ud, 0)
    current = current_dict.get((ud,id), 0)
    lam = 1/tau
    delta = latest - current

    return (1+(lam*delta))**(-1)

pezint = td.copy()
ts = pezint['created_at']
min_ts = min(ts[ts != 0])
max_ts = max(ts)
ts = pezint['created_at']
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
ts_scaled = min_max_scaler.fit_transform(ts)
pezint['created_at'] = ts_scaled
pezint = pezint.drop('interaction_type', 1)
pezint = pezint.groupby(['user_id', 'item_id']).aggregate(np.max).reset_index()
pezdict = dict(zip(zip(pezint['user_id'].values,pezint['item_id'].values),pezint['created_at'].values))

td['time_score'] = list(map(time_decay, td['user_id'].values, td['item_id'].values))
td = td.drop(['interaction_type','created_at'],1).drop_duplicates()

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# add users/items indexes columns
td['user_index'] = [users_dict.get(i) for i in td['user_id']]
td['item_index'] = [items_dict.get(i) for i in td['item_id']]

urm_time = sps.csr_matrix((td['time_score'], (td['user_index'], td['item_index'])))
t_urm = urm_time.T.copy()
similarity = metrics.cosine_similarity(t_urm, t_urm,dense_output=False)

diagonal = sps.diags(similarity.diagonal())
similarity = similarity - diagonal

co = t_urm.dot(urm_time)
diag_co = sps.diags(co.diagonal())
co = co - diag_co

sh_term = 3

k = co.copy()
k = k.power(0)

k = k * sh_term

d = co + k

n = co.multiply(similarity)

similarity = n.multiply(d.power(-1))