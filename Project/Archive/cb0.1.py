import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import Radond_sugg

K = 50
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
    suggestions = [0] * 10000
    for i in range(10000):
        user_id = userss[i]
        ratings = pd.DataFrame(columns=('item_interac', 'neighbor', 'n_rating'))
        itemss = interactions.loc[interactions['user_id'] == user_id]['item_id'].values
        for item_id in itemss:
            row = items[items['id'] == item_id]
            item = row[row.columns.difference(['id'])].values
            distances, indices = nbrs.kneighbors(item)
            similarities = 1 - distances
            for j in range(indices.size):
                index = indices[0,j]
                id = items.ix[index]['id'].astype(int)
                rui = score(user_id,id,similarities[0])
                if item_id != id:
                    list = [item_id, id, rui]
                    ratings.loc[j] = list
        suggestions[i] = ratings.sort_values('n_rating').tail(k)['neighbor'].values
        to_go = 9999 - i
        print('done ' + str(user_id)+', still to go: ' + str(to_go))

    return suggestions

def fill(matrixxx):
    users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
    result = np.column_stack((users, matrixxx))
    output(result)

def output(matrix):
    "Creates the file to be submitted given a kind of matrix"
    # output to file
    submission = open('submission2.csv', 'w')
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

