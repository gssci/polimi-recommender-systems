import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import sparse
import sklearn
from sklearn.neighbors import NearestNeighbors
import random
import scipy
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import project_utils as util #output, similarity function and soon other things
from sklearn.metrics.pairwise import cosine_similarity

K=50

#utility di sklearn che codifica attributi in forma di stringa, la usiamo per convertire le nazioni in numeri in modo coerente e confrontabile
le = preprocessing.LabelEncoder()

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
interactions = pd.read_csv('interactions.csv', delimiter="\t")

##FOR RATING ESTIMATION WE BASICALLY WANT TO COMPARE RECOMMENDABLE ITEMS TO
##ITEMS WITH KNOWN RATING

items_r = item_profiles[item_profiles['active_during_test'] == 1]

users = target_users['user_id'].values.tolist()
items = item_profiles.fillna(0)
items['country'] = items['country'].replace(0,'null')
items['country'] = le.fit_transform(items['country'])
to_encode = ['discipline_id','industry_id','country','region']
util.encode_feature(items,to_encode)
items = items.drop(to_encode,1)
items = items.drop(['title','tags','latitude','longitude', 'created_at','active_during_test','id'],1)
items = items.as_matrix()

ids = item_profiles['id'].values
ids_r = pd.DataFrame(item_profiles['id'])
ids = dict(zip(range(ids.size),ids))
###optimization related stuff###

items_r2 = set(items_r.reset_index().drop('index',1).index.tolist())
interactions = interactions.drop('created_at',1)
sommati = interactions.groupby(['user_id','item_id']).aggregate(np.sum).reset_index().values
keys = zip(sommati[:,0],sommati[:,1])
fast_int = dict(zip(keys,sommati[:,2]))
fast_interacted = fast_int.keys()


def work():

    ratings = np.zeros((10000,63446))

    for u in range(10000):
        user_id = users[u]

        rated_indices = ids_r[ids_r.id.isin(interacted)].index.tolist()
        raccomandabili = items_r2.difference(set(rated_indices))

        print(str(u))
        for i in list(raccomandabili):
            n = 0
            m = 0
            for j in rated_indices:
                s = 1 - distance.cosine(items[i],items[j])
                item_id = ids.get(j)
                n = n + s * fast_int.get((user_id,item_id))
                m = m + s

            ratings[u][i] = n/m

    return ratings

