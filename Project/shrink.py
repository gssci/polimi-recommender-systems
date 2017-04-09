import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter


target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("items_processed.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")
inter = interactions.drop("created_at", axis=1)


# all target_users not in interactions
ru_n = [u for u in target_users if u not in inter.values[:, 0]]
# all target_users in interactions
r_users = [u for u in target_users if u in inter.values[:, 0]]


# COLLABORATIVE FILTERING

# clean of interactions to create URM

# put all interaction_type values equal to 1
inter['interaction_type'] = 1

# drop duplicated rows
inter = inter.drop_duplicates()

# take lists of user ids and item ids
uid = pd.DataFrame(np.unique(inter['user_id']), columns=['user_id'])
uid.insert(1, 'user_index', uid.index.values)
iid = pd.DataFrame(np.unique(inter['item_id']), columns=['item_id'])
iid.insert(1, 'item_index', iid.index.values)

# add users and items indexes
inter = inter.merge(uid)
inter = inter.merge(iid)
inter = inter.drop_duplicates()


# UserRatingMatrix
urm = csr_matrix((inter['interaction_type'], (inter['user_index'], inter['item_index'])))


# indexes of r_users
ru_index = np.unique(inter[inter['user_id'].isin(r_users)]['user_index'].values).tolist()


# items active_during_test=0 ids
nact_items = items[items['active_during_test'] == 0]['id']
# items active_during_test=0 indexes
nact_index = np.unique(inter[inter['item_id'].isin(nact_items)]['item_index'].values)


# similarity matrix item-based
sim = cosine_similarity(urm.transpose(), dense_output=False)


l = len(r_users)
#for u in r_users:
u = r_users[0]

# items indexes of r_users u
index_r = np.unique(inter[inter['user_id'] == u]['item_index'].values)


# similarity matrix of index_r items
ind = sim[index_r].toarray()

# column of index_r indexes and inactive items are set equal to 0
ind[:, index_r] = 0
ind[:, nact_index] = 0


# matrix of similarity between r_item and other items
ns = (ind > 0).astype(float)

# sum all rows to have the number of similarity between user items and non-interacted items
ns = np.sum(ns, axis=0)

# sum all columns to have the similarity values between all user items and non-interacted items
ind = np.sum(ind, axis=0)


# weight on how many user items are similar non-interacted items (shrink = 3)
w = (ns*ind) / (ns + 3)

# items indexes in increasing order
index = w.argsort()

# number of items with similarity > 0
num_item = len([x for x in w if x > 0])

# takes only items indexes which have similarity > 0
index = index[-num_item:]

# similarity values in increasing order
w.sort()
# takes similarity values
s = w[-num_item:]

# item ids and indexes
id_index = inter[inter['item_index'].isin(index)][['item_id', 'item_index']].drop_duplicates()


l -= 1
print(l)
