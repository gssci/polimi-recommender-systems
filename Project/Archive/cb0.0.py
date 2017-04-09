import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import Radond_sugg


K = 5
k = 5

users = pd.read_csv("target_users.csv", delimiter="\t")
interactions_raw = pd.read_csv("interactions.csv", usecols=['user_id', 'item_id','interaction_type'], delimiter="\t")
items_raw = pd.read_csv("item_profile.csv", delimiter="\t")
items = pd.read_csv("item_profile.csv", usecols=['id','career_level', 'discipline_id', 'latitude', 'longitude','industry_id', 'employment', 'active_during_test'], delimiter="\t")
items = items[items['active_during_test'] == 1]
items = items[items.columns.difference(['active_during_test'])].reset_index()
items = items.fillna(0) #fills missing data with zeros

interactions = interactions_raw[interactions_raw['item_id'].isin(items['id'])].reset_index()

cols_to_norm = ['career_level', 'discipline_id', 'latitude', 'longitude','industry_id', 'employment']
items[cols_to_norm] = items[cols_to_norm].apply(lambda x: (x/ (x.max())))

def S(i, j):
    H = 5
    a = items[items['id'] == i]
    ii = a[a.columns.difference(['id'])].values
    b = items[items['id'] == j]
    jj = b[b.columns.difference(['id'])].values
    return cosine_similarity(ii,jj)






# items[cols_to_norm] = items[cols_to_norm].apply(lambda x: (x / (x.max())))
#
X = items[items.columns.difference(['id'])].values
nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto', metric='euclidean').fit(X)



userss = [val for sublist in users.values for val in sublist]

def inter_type(user,item):
    inter = interactions[interactions['user_id']==user]
    try:
        w = inter[inter['item_id'] == item]['interaction_type'].values.max()
        return w
    except ValueError:
        return 1

def score(user,item,sim_matrix):
    num = 0
    den = 0
    for s in sim_matrix:
        num = num + inter_type(user,item) * s
        den = den + s
    r = num / den
    return r

def rateall():
    for user_id in userss:
        ratings = pd.DataFrame(columns=('user','item_interac', 'neighbor', 'n_rating'))
        itemss = interactions.loc[interactions['user_id'] == user_id]['item_id'].values
        i = 0
        for item_id in itemss:
            row = items[items['id'] == item_id]
            item = row[row.columns.difference(['id'])].values
            distances, indices = nbrs.kneighbors(item)
            similarities = 1 - distances
            for j in range(indices.size):
                index = indices[0,j]
                id = items.loc[index]['id'].astype(int)
                rui = score(user_id,id,similarities)
                list = [user_id, item_id, id, rui]
                print(list)
                ratings.loc[i] = list
                i = i + 1

    return ratings




#
# #     max = np.amax(numbers)
# #     for x in numbers:
# #         x = np.divide(x,max)
# #         print(x)
# #
# # X = items[items.columns.difference(['id'])]
# #nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='euclidean').fit(X.values)
#
#
# def norm_string(x):
#     numbers = [int(n) for n in x.split(',')]
#     return np.average(numbers)
