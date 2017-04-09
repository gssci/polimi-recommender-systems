from __future__ import division
import numpy as np
import pandas as pd
import nltk
from scipy import sparse as sps

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

items = pd.read_csv("item_profile.csv", delimiter='\t')
attivipd = items[items['active_during_test'] == 1]
items_nact_ids = items[items['active_during_test'] == 0]['id'].drop_duplicates().values

target_users = pd.read_csv("target_users.csv", delimiter='\t')
users = target_users['user_id'].values

ids = items['id'].values
index_to_ids = dict(zip(range(ids.size), ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}

items_attivi_indices = [ids_to_index.get(i) for i in attivipd['id'].values]

index_to_uds = dict(zip(range(users.size), users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

user_profile = pd.read_csv("user_profile.csv", delimiter='\t')
uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size), uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

items = items[['id', 'tags']]
items = items.fillna(0)
items = items[(items['tags'] != 0) & (items['tags'] != '0')]

def spit(s):
    return s.split(',')

corpus = items['tags'].ravel()
alltags = [tag for tags in list(map(spit, corpus)) for tag in tags]
fdist = nltk.FreqDist(alltags)
tags = [k for k in fdist.keys() if fdist.get(k) > 1]
tags = np.array(tags).astype(int)

data = pd.concat([pd.Series(row['id'], row['tags'].split(',')) for _, row in items.iterrows()]).reset_index()
data = data[data.columns[::-1]].rename(index=str, columns={"index": "tag", 0: "item_id"})
data = data.reset_index().drop('index', 1)
data['tag'] = data['tag'].astype(int)
data = data[data.tag.isin(tags)]
# data = pd.read_csv("tags.csv", delimiter='\,',engine='python').drop('Unnamed: 0',1)

tags = np.unique(data['tag'].values)
tag_to_index = dict(zip(tags, range(len(tags))))

data['y'] = 1
data['item_index'] = [ids_to_index.get(i) for i in data['item_id']]
data['tag_index'] = [tag_to_index.get(i) for i in data['tag']]

icm = sps.csr_matrix((data['y'], (data['item_index'], data['tag_index'])))
icm = icm.astype(bool).astype(int)

td = pd.read_csv('interactions.csv', delimiter="\t")

def interagiti(user_id):
    return np.unique(td[td['user_id'] == user_id]['item_id'].values)


ICM = icm[items_attivi_indices]
all_cont = list(np.array(ICM.sum(axis=1)).ravel())

R_out = np.zeros((10000, 167956))

remaining = len(users)
confint2 = pd.read_csv('interactions.csv', delimiter='\t')
confint2 = confint2.drop('created_at', 1)
confint2 = confint2.groupby(['user_id', 'item_id']).aggregate(np.sum).reset_index()
T = dict(zip(zip(confint2['user_id'].values, confint2['item_id'].values), confint2['interaction_type'].values))

timeint = pd.read_csv('interactions.csv', delimiter="\t")
times = timeint.sort_values(['user_id', 'created_at'], ascending=[True, False])
times = times.groupby(['user_id','item_id']).aggregate(np.max).reset_index().sort_values(['user_id','created_at'],ascending=[True,False])

latest_grp = times[['user_id','created_at']].groupby('user_id').first().reset_index()

latest_dict = dict(zip(latest_grp['user_id'].values,latest_grp['created_at'].values))

current_dict = dict(zip(zip(times['user_id'].values,times['item_id'].values),times['created_at'].values))

def time_decay(ud,id):
    tau = float(60 * 60 * 24 * 7) #month in seconds
    latest = latest_dict.get(ud, 0)
    current = current_dict.get((ud,id), 0)
    lam = 1/tau
    delta = latest - current

    return (1+(lam*delta))**(-1)

def t(u,i):
    x = T.get((u,i),0)
    return np.log(1+x)

for u in users:
    seen = interagiti(u)

    if seen.size > 0:
        seen_indices = [ids_to_index.get(s) for s in seen]
        curr = icm[seen_indices].copy()
        curr = curr.astype(float)
        confidence_scores = np.array(list(map(t, np.zeros(seen.size) + u, seen)))
        confidence_scores = np.zeros(curr.shape) + confidence_scores.reshape(len(confidence_scores), 1)
        time_scores = np.array(list(map(time_decay, np.zeros(seen.size) + u, seen)))
        time_scores = np.zeros(curr.shape) + time_scores.reshape(len(time_scores), 1)
        curr = curr.multiply(time_scores)
        curr = curr.multiply(confidence_scores)
        curr_intersection = curr.dot(ICM.T)

        U = curr_intersection.copy()#.power(0)
        Ut = U.T.copy()
        rated_cont = list(np.array(curr.sum(axis=1)).ravel())
        U = U.dot(sps.diags(all_cont))
        Ut = Ut.dot(sps.diags(rated_cont))
        U = U + Ut.T
        U = U - curr_intersection
        sim = curr_intersection.multiply(U.power(-1))

        rec = np.zeros(167956)
        rec[items_attivi_indices] = np.array(sim.sum(axis=0)).ravel()
        rec[seen_indices] = 0
        rec = np.true_divide(rec, np.max(rec))
        R_out[uds_to_index.get(u)] = rec

    remaining -= 1
    print(str(remaining))

save_sparse_csr('CBF_tags_ALL1',sps.csr_matrix(R_out[0:5000]))
save_sparse_csr('CBF_tags_ALL2',sps.csr_matrix(R_out[5000:10000]))