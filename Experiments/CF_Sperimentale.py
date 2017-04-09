from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
from scipy import sparse

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
userss = target_users['user_id'].values

item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
ids = item_profiles['id'].values
items_inattivi = item_profiles[item_profiles['active_during_test'] == 0]['id'].values

index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(userss.size),userss))
uds_to_index = {v: k for k, v in index_to_uds.items()}

interactions = pd.read_csv('interactions.csv', delimiter="\t")
#interactions = interactions[interactions['interaction_type'] == 1]
observations = gl.SFrame(interactions)

model6 = gl.recommender.item_similarity_recommender.create(observations, user_id='user_id', item_id='item_id', target=None,
                                                               user_data=None, item_data=None, similarity_type='cosine', only_top_k=1700, verbose=True)
R7 = sparse.lil_matrix((10000,167956))

td = pd.read_csv('interactions.csv', delimiter="\t")

def interagiti(user_id):
    return np.unique(td[td['user_id'] == user_id]['item_id'].values)

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

confint = pd.read_csv('interactions.csv', delimiter='\t')
confint = confint.drop('created_at', 1)
confint = confint[confint['interaction_type']==1]
confint = confint.groupby(['user_id', 'item_id']).aggregate(np.sum).reset_index()
T = dict(zip(zip(confint['user_id'].values, confint['item_id'].values), confint['interaction_type'].values))

def t(u,i):
    x = T.get((u,i),0)
    return np.log2(1+x)

remaining = 10000
for u in userss:
    js = interagiti(u)

    if js.size > 0:

        suggestions = model6.get_similar_items(list(js),k=1000).to_dataframe()

        y = suggestions['item_id'].values

        suggestions = suggestions.drop('item_id',1)

        #confidence = map(t,(np.zeros(y.size) + u), y)
        timesw = map(time_decay, (np.zeros(y.size) + u), y)

        suggestions['score'] *= timesw
        #suggestions['score'] *= confidence

        suggestions = suggestions.groupby('similar').aggregate(np.sum).reset_index()
        #suggestions = suggestions[~suggestions.similar.isin(js)]

        R7[uds_to_index.get(u), [ids_to_index.get(k) for k in suggestions['similar'].values]] = suggestions['score'].values

    remaining -= 1
    print(str(remaining))

save_sparse_csr('R7_ALL',R7.tocsr())
print("Saved R7_ALL")
