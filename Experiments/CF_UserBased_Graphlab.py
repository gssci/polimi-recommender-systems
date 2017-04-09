from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import random
import time
from scipy import sparse
from timedecay import pezdict
from docutils.nodes import transition
import math

#interactions are not all items that can be recommended
user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("user_profile.csv", delimiter='\t')
userss = user_profiles['user_id'].values

items = gl.SFrame.read_csv('item_profile.csv', delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
ids = item_profiles['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}

items_inattivi = item_profiles[item_profiles['active_during_test'] == 0]['id'].values
itemsss = pd.read_csv("item_profile.csv", delimiter='\t')
target_users = pd.read_csv("user_profile.csv", delimiter='\t')
ids = itemsss['id'].values
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
index_to_uds = dict(zip(range(userss.size),userss))
uds_to_index = {v: k for k, v in index_to_uds.items()}

interactions = pd.read_csv('interactions.csv', delimiter="\t")

def interagiti(user_id):
    return np.unique(interactions[interactions['user_id'] == user_id]['item_id'].values)

def pez_decay(ud,id):
    return pezdict.get((ud,id),0)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
            indptr =array.indptr, shape=array.shape )

interactions['time'] = list(map(pez_decay, interactions['user_id'].values, interactions['item_id'].values))

observations = gl.SFrame(interactions)

#dio = gl.recommender.item_content_recommender.create(items, 'item_id', observation_data=observations, user_id="user_id", target="interaction_type")
six_most_popular = [2778525,1244196,1386412,657183,2791339,536047]
most_popular = [2778525, 1386412, 1244196, 657183, 2791339]

def fill(input):
    suggestions = [0] * 10000
    for i in range(10000):
        suggestions[i] = input[input['user_id'] == userss[i]]['item_id'].values
    return suggestions

def fill2(rec):
    suggestions = [0] * 10000

    for i in range(10000):
        sss = rec[rec['user_id'] == userss[i]].sort_values('score',ascending=False).head(5)['item_id'].values
        j = 0

        while sss.size < 5:
            sss = np.append(sss,[most_popular[j]])
            j = j+1

        suggestions[i] = sss
    return suggestions


model6 = gl.recommender.item_similarity_recommender.create(observations, user_id='user_id', item_id='item_id', target=None,
                                                               user_data=None, item_data=None, similarity_type='cosine', only_top_k=1700, verbose=True)


#items = gl.toolkits.feature_engineering.AutoVectorizer(column_interpretations={"title":"short_text","career_level":"categorical","discipline_id":"categorical","industry_id":"categorical","region":"categorical"}).fit_transform(items)

#result = np.column_stack((userss,s))

#model666 = gl.recommender.item_content_recommender.create(items, item_id='item_id', observation_data=training_data, user_id='user_id')
#AC = gl.toolkits.feature_engineering.AutoVectorizer(column_interpretations={"title":"short_text","discipline_id":"categorical","industry_id":"categorical","region":"categorical"})

#user_data = gl.SFrame.read_csv("user_profile.csv", delimiter='\t')
#model7 = gl.recommender.factorization_recommender.create(training_data,user_id="user_id",item_id="item_id",target='interaction_type',user_data=user_data,item_data=items)

def weight(i):
    return 1 - 1/i**(1/2)

aja = pd.read_csv('interactions.csv', delimiter="\t")
clicks = aja[aja['interaction_type'] == 1]
clicks = clicks.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type', ascending=False).drop('user_id',1)
ws = dict(zip(clicks['item_id'].values,map(weight,clicks['interaction_type'].values)))

def w(i):
    return ws.get(i,0)


def output(matrix):
    """Matrix must be be of shape (10000,6) first column is for user_id, the remaining five are the id's of the recommended items"""
    submission = open('test_submission.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in matrix:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
    return

def fillpop(sugg):
    interaction = pd.read_csv('interactions.csv', delimiter="\t")
    a = np.unique(interaction['user_id'].values)
    anonimi = [u for u in userss if u not in a]
    uds = target_users['user_id'].values
    uds_to_index = dict(zip(uds, range(uds.size)))

    indexeses = [uds_to_index.get(k) for k in anonimi]

    for i in indexeses:
        #sugg[i] = np.array(random.sample(six_most_popular,5))
        sugg[i] = most_popular
    return

recs = model6.recommend(userss,k=1700,exclude_known=False)
rec = recs.to_dataframe()

keys = rec['item_id'].values
wss = [w(k) for k in keys]
wss = np.array(wss)
rec['score'] *= wss
rec = rec.sort_values(['user_id','score'],ascending=[True,False])

RX = sparse.lil_matrix((40000,167956))

# utenti = rec['user_id'].values
# remaining = len(utenti)
indicator = 1
start = 0
end = 1700
for u in userss:
    stuff = rec.iloc[start:end]
    row = stuff['score'].values

    if len(row) > 0:
        row = np.true_divide(row,np.max(row))

    RX[[uds_to_index.get(v) for v in stuff['user_id'].values],[ids_to_index.get(i) for i in stuff['item_id'].values]] = row
    indicator += 1
    start += 1700
    end += 1700
    print(str(indicator))

save_sparse_csr('CF_ItemBased_GL_ALL',RX.tocsr())
print("done")

# save_sparse_csr('RX1',RX1.tocsr())
# print("saved RX1")
# # 
confint2 = pd.read_csv('interactions.csv', delimiter='\t')
confint2 = confint2.drop('created_at', 1)
confint2 = confint2.groupby(['user_id', 'item_id']).aggregate(np.sum).reset_index()
T = dict(zip(zip(confint2['user_id'].values, confint2['item_id'].values), confint2['interaction_type'].values))

def t(u,i):
    x = T.get((u,i),0)
    return np.log(1+x)
# 
# timeint = pd.read_csv('interactions.csv', delimiter="\t")
# times = timeint.sort_values(['user_id', 'created_at'], ascending=[True, False])
# times = times.groupby(['user_id','item_id']).aggregate(np.max).reset_index().sort_values(['user_id','created_at'],ascending=[True,False])
# 
# latest_grp = times[['user_id','created_at']].groupby('user_id').first().reset_index()
# 
# latest_dict = dict(zip(latest_grp['user_id'].values,latest_grp['created_at'].values))
# 
# current_dict = dict(zip(zip(times['user_id'].values,times['item_id'].values),times['created_at'].values))
# 
# def time_decay(ud,id):
#     tau = float(60 * 60 * 24 * 7) #month in seconds
#     latest = latest_dict.get(ud, 0)
#     current = current_dict.get((ud,id), 0)
#     lam = 1/tau
#     delta = latest - current
# 
#     return (1+(lam*delta))**(-1)
# 
# def pez_decay(ud,id):
#     return pezdict.get((ud,id),0)
# 
# R7 = sparse.lil_matrix((40000,167956))
# remaining = 40000
# 
# interactionss = pd.read_csv('interactions.csv', delimiter="\t")
# def interagiti(user_id):
#     return np.unique(interactionss[interactionss['user_id'] == user_id]['item_id'].values)
# 
# for u in userss:
#     js = interagiti(u)
# 
#     if js.size > 0:
#         suggestions = model6.get_similar_items(list(js),k=1700).to_dataframe().drop('rank',1)
# 
#         y = suggestions['item_id'].values
# 
#         suggestions = suggestions.drop('item_id',1)
#         # #confidence = map(t,(np.zeros(y.size) + u), y)
#         # timesw = map(pez_decay, (np.zeros(y.size) + u), y)
#         #
#         # suggestions['score'] *= timesw
#         # #suggestions['score'] *= confidence
# 
#         suggestions = suggestions.groupby('similar').aggregate(np.sum).sort_values('score', ascending=False).reset_index()
#         suggestions = suggestions[~suggestions.similar.isin(js)]
#         rec = suggestions['score'].values
#         if len(rec) > 0:
#             rec = np.true_divide(rec,np.max(rec))
#         R7[uds_to_index.get(u), [ids_to_index.get(k) for k in suggestions['similar'].values]] = rec
# 
#     remaining -= 1
#     print(str(remaining))
# 
# save_sparse_csr('CF_II',R7.tocsr())
# print("saved R7")