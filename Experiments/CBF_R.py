from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse

items0 = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
users = target_users['user_id'].values
ids = items0['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

to_evaluate = ['discipline_id','industry_id','employment','career_level']

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    return


td = pd.read_csv('training_data.csv', delimiter="\t")
td = td.drop('Unnamed: 0', 1)

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

confint = pd.read_csv('training_data.csv ', delimiter='\t')
confint = confint.drop('created_at', 1)
confint = confint[confint['interaction_type']==1]
confint = confint.groupby(['user_id', 'item_id']).aggregate(np.sum).reset_index()
T = dict(zip(zip(confint['user_id'].values, confint['item_id'].values), confint['interaction_type'].values))

def t(u,i):
    x = T.get((u,i),0)
    return np.log2(1+x)

for cat in to_evaluate:
    RA = sparse.lil_matrix((10000, 167956))
    remaining = 10000
    items = pd.read_csv("item_profile.csv", delimiter='\t')
    items = items[['id',cat]]
    items[cat] = items[cat].fillna(0)
    items = items[items[cat]!=0]
    items[cat] = items[cat].astype(int)
    data = gl.SFrame(items)
    data = data.dropna()

    model = gl.recommender.item_similarity_recommender.create(data,user_id=cat,item_id='id',similarity_type='jaccard')

    for u in users:
        js = interagiti(u)
        n = js.size
        if n > 0:
            suggestions = model.get_similar_items(list(js),k=100).to_dataframe().drop('rank',1)
            y = suggestions['id'].values
            suggestions = suggestions.drop('id',1)

            confidence = map(t, (np.zeros(y.size) + u), y)
            suggestions['score'] *= confidence

            timesw = map(time_decay, (np.zeros(y.size) + u), y)
            suggestions['score'] *= timesw

            suggestions = suggestions.groupby('similar').aggregate(np.sum).reset_index()
            suggestions = suggestions[~suggestions.similar.isin(js)]

            RA[uds_to_index.get(u), [ids_to_index.get(k) for k in suggestions['similar'].values]] = suggestions['score'].values
        remaining -= 1
        print(str(remaining))

    save_sparse_csr(('R'+cat),RA.tocsr())


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
#
# import R
# import map5
# for cat in to_evaluate:
#     R.makeR(load_sparse_csr('R'+cat+'.npz'),'R'+cat)
#     map5.evaluate_submission('R'+cat+'.csv')