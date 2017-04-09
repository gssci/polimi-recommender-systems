import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import scipy.sparse as sps
from sklearn.preprocessing import MinMaxScaler

target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("items_processed.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")
inter = interactions.drop("created_at", axis=1)


# all target_users not in interactions
ru_n = [u for u in target_users if u not in inter.values[:, 0]]
# all target_users in interactions
r_users = [u for u in target_users if u in inter.values[:, 0]]


# WEIGHTS

def weight(i):
    return 1 - 1/i**(1/2)

interact = interactions.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type', ascending=False).drop('user_id', 1)
ws = dict(zip(interact['item_id'].values, np.apply_along_axis(weight, 0, interact['interaction_type'].values)))

def w(i):
    x = 0.01
    try:
        x = ws.get(i)
    except KeyError:
        x = 0.01
    return x


# TOP-POPULAR

tpop = []
# top-popular for users not in interactions
def topop():
    most_popular = [2778525, 1244196, 1386412, 657183, 2791339]

    for i in range(len(ru_n)):
        np.random.shuffle(most_popular)    # random order
        tpop.append([ru_n[i]]+most_popular)
        print(i)


# COLLABORATIVE FILTERING

# create matrix inter: user_id, user_index, item_id, item_index, interaction_type

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
nact_ids = items[items['active_during_test'] == 0]['id']
# items active_during_test=0 indexes
nact_index = np.unique(inter[inter['item_id'].isin(nact_ids)]['item_index'].values)


# weights the most clicked
sort_id = inter[['item_index','item_id']].drop_duplicates().sort_values('item_index')['item_id']
weights = list(map(w, sort_id))


# ITEM-BASED CF

# similarity matrix item-based
sim = cosine_similarity(urm.transpose(), dense_output=False)

# submission matrix
subm = []

def cf():

    l = len(r_users)

    for u in r_users:
        #u = r_users[0]

        # items indexes of r_users u
        index_r = np.unique(inter[inter['user_id'] == u]['item_index'].values)

        # similarity matrix of index_r items
        ind = sim[index_r].toarray()

        # column of index_r indexes and inactive items are set equal to 0
        ind[:, index_r] = 0
        ind[:, nact_index] = 0

        # sum all rows
        ind = np.sum(ind, axis=0)

        # weights for most clicked items
        #ind = np.multiply(ind, weights)

        # take indexes of 5 most similar items
        indexes = ind.argsort()[-5:][::-1]

        # extract item part of inter data frame order by item indexes
        inter_item = inter[['item_id', 'item_index']].drop_duplicates().sort_values('item_index').set_index('item_index')

        # takes item_ids related to indexes
        ids = inter_item['item_id'].values[indexes].tolist()

        # insert user-id and item-ids in submission matrix
        subm.append([u] + ids)

        l -= 1
        print(l)
        break


# USER-BASED CF

# similarity matrix user-based
usim = cosine_similarity(urm, dense_output=False).toarray()

# diagonal equal to 0
np.fill_diagonal(usim, 0)

# submission matrix
usubm = []
matr = sps.lil_matrix((len(target_users), urm.shape[1]))
def cfu():
    l = len(ru_index)
    for ui in ru_index:
        # ui = ru_index[0]

        # id of ui user
        usid = np.unique(inter[inter['user_index'] == ui]['user_id'].values).tolist()

        # ids of items rated by user ui
        iid = np.unique(inter[inter['user_index'] == ui]['item_id'].values).tolist()

        # sim matrix for ui r_users
        rusim = usim[ui]

        # index of sorted rusim values
        us_ind = rusim.argsort()

        # sorted rusim
        rusim.sort()

        # takes only user indexes which have similarity greater than 0
        nu = [u for u in rusim if u > 0]
        us_ind = us_ind[-len(nu):]

        # ids of items with which users us_ind have interacted
        iin = inter[inter['user_index'].isin(us_ind)]['item_id'].tolist()

        # remove nonactive items and just interacted
        iin = [i for i in iin if i not in nact_ids]
        iin = [i for i in iin if i not in iid]

        # count occurence of ids in iin
        icount = Counter(iin)

        # take most common 5
        ids = icount.most_common(5)
        ids = [i[0] for i in ids]

        usubm.append(usid + ids)

        l -= 1
        print(l)


# submission csv file
def output(matrix):
    # matrix of shape (10000,6): 1st column is for user_id, the remaining five are the ids of the recommended items

    submission = open('submission.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in matrix:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line += str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
