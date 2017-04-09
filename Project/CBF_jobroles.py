from __future__ import division
import numpy as np
import pandas as pd
import nltk
from scipy import sparse as sps

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

items = pd.read_csv("item_profile.csv", delimiter='\t')
user_profile = pd.read_csv("user_profile.csv", delimiter='\t')
users = pd.read_csv("user_profile.csv", delimiter='\t')
all_users_id = user_profile['user_id'].values
target_users = pd.read_csv('target_users.csv', delimiter='\t')['user_id'].values

ids = items['id'].values
index_to_ids = dict(zip(range(ids.size), ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}

items_nact_ids = items[items['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [ids_to_index.get(i) for i in items_nact_ids]

index_to_uds = dict(zip(range(all_users_id.size), all_users_id))
uds_to_index = {v: k for k, v in index_to_uds.items()}

user_profile = user_profile[['user_id', 'jobroles']]
user_profile = user_profile.fillna(0)
user_profile = user_profile[(user_profile['jobroles'] != 0) & (user_profile['jobroles'] != '0')]

def spit(s):
    return s.split(',')

corpus = user_profile['jobroles'].ravel()
alljobroles = [tag for jobroles in list(map(spit, corpus)) for tag in jobroles]
fdist = nltk.FreqDist(alljobroles)
jobroles = [k for k in fdist.keys() if fdist.get(k) > 1]
jobroles = np.array(jobroles).astype(int)

data = pd.concat([pd.Series(row['user_id'], row['jobroles'].split(',')) for _, row in user_profile.iterrows()]).reset_index()
data = data[data.columns[::-1]].rename(index=str, columns={"index": "tag", 0: "user_id"})
data = data.reset_index().drop('index', 1)
data['tag'] = data['tag'].astype(int)
data = data[data.tag.isin(jobroles)]

jobroles = np.unique(data['tag'].values)
jobrole_to_index = dict(zip(jobroles, range(len(jobroles))))

data['y'] = 1
data['user_index'] = [uds_to_index.get(i) for i in data['user_id']]
data['jobrole_index'] = [jobrole_to_index.get(i) for i in data['tag']]

ucm = sps.csr_matrix((data['y'], (data['user_index'], data['jobrole_index'])))
ucm = ucm.astype(bool).astype(int)

all_cont = list(np.array(ucm.sum(axis=1)).ravel())

R_out = sps.lil_matrix((40000, 167956))

interactions = pd.read_csv("training_data.csv", delimiter="\t")
inter = interactions.copy()
inter = inter.drop(['interaction_type', 'created_at'], 1)
inter = inter.groupby(['user_id', 'item_id']).size().reset_index()
inter = inter.rename(columns={0: 'count'})
inter['count'] = 1

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'].values, users.index))
items_dict = dict(zip(items['id'].values, items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict

# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]


# URM : user-rating matrix
urm = sps.csr_matrix((inter['count'], (inter['user_index'], inter['item_index'])))


remaining = len(target_users)

for u in target_users:
    curr = ucm[uds_to_index.get(u)].copy()
    curr = curr.astype(float)
    curr_intersection = curr.dot(ucm.T)

    U = curr_intersection.copy().power(0)
    Ut = U.T.copy()
    rated_cont = list(np.array(curr.sum(axis=1)).ravel())
    U = U.dot(sps.diags(all_cont))
    Ut = Ut.dot(sps.diags(rated_cont))
    U = U + Ut.T
    U = U - curr_intersection

    sim = curr_intersection.multiply(U.power(-1))

    rec = sim.dot(urm)
    rec = rec.toarray().ravel()
    rec[items_nact] = 0
    rec = np.true_divide(rec, np.max(rec))

    R_out[uds_to_index.get(u)] = rec

    remaining -= 1
    print(str(remaining))

save_sparse_csr('CBF_jobroles', sps.csr_matrix(R_out))
print("==::complete::==")