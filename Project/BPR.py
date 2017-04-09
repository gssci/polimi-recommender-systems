import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity
import random


target_users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
users = pd.read_csv("user_profile.csv", delimiter="\t")
items = pd.read_csv("items_processed.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", delimiter="\t")


# PREPROCESSING

# cleaning interactions
inter = interactions.drop('created_at', 1)
inter = inter[inter['interaction_type'] == 1]
inter = inter.drop_duplicates()

# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))


# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]


# URM : user-rating matrix
urm = sps.csr_matrix((inter['interaction_type'], (inter['user_index'], inter['item_index'])))


# BPR : bayesian personalized ranking

# similarity matrix
sim = cosine_similarity(urm.transpose(), dense_output=False)


# LEARN-BPR

# # 1 : initialize theta as a copy of sim
# theta = sim.copy().tolil()
# alpha = 0.05
# lambda_t = 10 ^ -8
#
# sum = 0
# c = 0
# while c < 1000:
#     # 2 : random user's index u and items indexes i and j
#     u = users_dict.get(random.choice(target_users))
#
#     # items indexes with which user u has interacted
#     items_u = inter[inter['user_index'] == u]['item_index'].drop_duplicates().values
#     # check if user u has interacted with at least 2 items
#     if len(items_u) > 1:
#         # items indexes with which user u has not interacted
#         items_nu = list(set(items_dict.values()) - set(items_u))
#         # random item's index i from items_u
#         i = random.choice(items_u)
#         # random item's index j from items_nu
#         j = random.choice(items_nu)
#
#
#         # 3 : update theta
#
#         # compute xui : how much item i is similar to others items_u
#         items_uni = list(set(items_u) - set([i]))
#         xui = np.sum(sim[items_uni], axis=0)[0, i]
#
#         # compute xuj : how much item j is similar to items_u
#         xuj = np.sum(sim[items_u], axis=0)[0, j]
#
#         # compute xuij
#         xuij = xui - xuj
#
#         if xuij > 0:
#             incr = alpha*(np.exp(-xuij)/(1 - np.exp(-xuij))*1 + lambda_t*theta[i, j])
#             # update theta in [i,j] -> derivative = 1
#             theta[i, j] += incr
#             # update theta in [j,i] -> derivative = -1
#             theta[j, i] += alpha*(np.exp(-xuij)/(1 - np.exp(-xuij))*(-1) + lambda_t*theta[j, i])
#
#             sum += incr
#
#     c += 1
#     if c == 1000:
#         if sum < 0.01:
#             break
#         else:
#             sum = 0
#             c = 0
#
#     print(sum)
#     print(c)
#
#


def clean_output():

    bprsubm = pd.read_csv("output.csv", delimiter="\t", header=None)

    most_popular = [2778525, 1244196, 1386412, 657183, 2791339]

    subm = []

    for i in range(10000):

        u = bprsubm.iloc[i][0]
        s = bprsubm.iloc[i][1]

        if s != '[]':
            s = s.replace(']', '')
            s = s.replace('[', '')

            s = s.split(',')
            s = list(map(lambda x: int(x.split(':')[0]), s))

            subm.append([u]+s)
        else:
            subm.append([u] + most_popular)

    return subm


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
