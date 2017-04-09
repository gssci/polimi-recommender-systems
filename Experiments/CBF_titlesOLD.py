from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse
from timedecay import pezdict

items = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
users = target_users['user_id'].values
ids = items['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

user_profile = pd.read_csv("user_profile.csv", delimiter='\t')
uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

# items = items[['id','title']]
# items = items.fillna(0)
# items = items[(items['title'] != 0) & (items['title'] != '0')]
# 
# def spit(s):
#     return s.split(',')
# 
# corpus = items['title'].ravel()
# alltags = [tag for tags in list(map(spit,corpus)) for tag in tags]
# fdist = nltk.FreqDist(alltags)
# tags = [k for k in fdist.keys() if fdist.get(k) > 1]
# tags = np.array(tags).astype(int)
# 
# data = pd.concat([pd.Series(row['id'], row['title'].split(',')) for _, row in items.iterrows()]).reset_index()
# data = data[data.columns[::-1]].rename(index=str, columns={"index": "title", 0: "item_id"})
# data = data.reset_index().drop('index',1)
# data['title'] = data['title'].astype(int)
# data = data[data.title.isin(tags)]
# data = data.reset_index().drop('index',1)
# 
# #data = pd.read_csv('title.csv', delimiter=',').drop('Unnamed: 0',1)
# 
# data = gl.SFrame(data)
# model = gl.recommender.item_similarity_recommender.create(data,user_id='title',item_id='item_id',similarity_type='jaccard')
model = gl.load_model('TITLEMODEL')

RA = sparse.lil_matrix((40000,167956))
remaining = len(uds2)

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

def pez_decay(ud,id):
    return pezdict.get((ud,id),0)


confint = pd.read_csv('interactions.csv', delimiter='\t')
confint = confint.drop('created_at', 1)
confint = confint[confint['interaction_type']==1]
confint = confint.groupby(['user_id', 'item_id']).aggregate(np.sum).reset_index()
T = dict(zip(zip(confint['user_id'].values, confint['item_id'].values), confint['interaction_type'].values))

def t(u,i):
    x = T.get((u,i),0)
    return np.log2(1+x)

for u in users:
    js = interagiti(u)
    n = js.size

    if n > 0:
        suggestions = model.get_similar_items(list(js),k=1000).to_dataframe().drop('rank',1)
        y = suggestions['item_id'].values.ravel()
        suggestions = suggestions.drop('item_id',1)

        suggestions = suggestions.groupby('similar').aggregate(np.sum).sort_values('score', ascending=False).reset_index()
        suggestions = suggestions[~suggestions.similar.isin(js)]

        rec = suggestions['score'].values
        if len(rec) > 0:
            rec = np.true_divide(rec,np.max(rec))
            RA[uds_to_index2.get(u), [ids_to_index.get(k) for k in suggestions['similar'].values]] = rec
        print(str(remaining))
    remaining -= 1

print('done')

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

save_sparse_csr('CBF_titles_ALL',RA.tocsr())
