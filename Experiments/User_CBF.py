from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import nltk
from scipy import sparse
from sklearn import preprocessing
abs = preprocessing.MaxAbsScaler()
items0 = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
interactions = pd.read_csv('training_data.csv', delimiter='\t')
user_profile = pd.read_csv('user_profile.csv', delimiter='\t')
di = {'de': 1, "at": 2, "ch":3, "non_dach":0}
user_profile = user_profile.replace({'country':di})
user_profile = user_profile.fillna(0)
users = target_users['user_id'].values
ids = items0['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(users.size),users))
uds_to_index = {v: k for k, v in index_to_uds.items()}

uds2 = user_profile['user_id'].values
index_to_uds2 = dict(zip(range(uds2.size),uds2))
uds_to_index2 = {v: k for k, v in index_to_uds2.items()}

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

anonimi = [u for u in users if interagiti(u).size<=0]
anonimi_indices = [uds_to_index.get(a) for a in anonimi]

# us = user_profile[['user_id','edu_fieldofstudies']]
# us = us.fillna(0)
# us = us[(us['edu_fieldofstudies']!=0) & (us['edu_fieldofstudies'] !='0')]
# 
# def spit(s):
#     return s.split(',')
# 
# corpus = us['edu_fieldofstudies'].ravel()
# allroles = [tag for tags in list(map(spit,corpus)) for tag in tags]
# fdist = nltk.FreqDist(allroles)
# tags = [k for k in fdist.keys() if fdist.get(k) > 1]
# tags = np.array(tags).astype(int)
# 
# data = pd.concat([pd.Series(row['user_id'], row['edu_fieldofstudies'].split(',')) for _, row in us.iterrows()]).reset_index()
# data = data[data.columns[::-1]].rename(index=str, columns={"index": "jobrole", 0: "user_id"})
# data = data.reset_index().drop('index',1)
# data['jobrole'] = data['jobrole'].astype(int)
# data = data[data.jobrole.isin(tags)]
# #data = pd.read_csv("tags.csv", delimiter='\t').drop('Unnamed: 0',1)
# data = gl.SFrame(data)
# 
# model = gl.recommender.item_similarity_recommender.create(data,user_id='jobrole',item_id='user_id',similarity_type='jaccard')
# model.save('./edu_fieldofstudies')

model = gl.load_model('edu_fieldofstudies')
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    return

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


ALS = load_sparse_csr('ALS4k.npz')
ALS = abs.fit_transform(ALS.T).T

R_sorgente = ALS
R_out = sparse.lil_matrix((10000,167956))
remaining = np.size(users)

sims = model.get_similar_items(list(users),k=500)
similar = sims.to_dataframe()

for a in users:
    simili = similar[similar['user_id'] == a]
    rec = np.dot(R_sorgente[[uds_to_index2.get(v) for v in simili['similar'].values]].toarray().T,
                                        simili['score'].values)
    cols = np.argsort(rec)[::-1][:1000]
    # 
    # if len(rec)>0:
    #     rec = np.true_divide(rec,np.max(rec))

    R_out[uds_to_index.get(a),cols] = rec[cols]
    remaining -= 1
    print(str(remaining))

save_sparse_csr('R_edu_fieldofstudies',R_out.tocsr())
print("done")