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
import scipy.sparse as sps

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
items_fighi = items[items['active_during_test'] == 1]
items = items.drop(['title','tags','latitude','longitude', 'created_at','active_during_test','id'],1)
items_fighi = items_fighi.drop(['title','tags','latitude','longitude', 'created_at','active_during_test','id'],1)
items = items.as_matrix()

ids = item_profiles['id'].values
ids_r = pd.DataFrame(item_profiles['id'])
index_to_ids = dict(zip(range(ids.size),ids))
ids_to_index = {v: k for k, v in index_to_ids.items()}
###optimization related stuff###

items_r2 = set(items_r.reset_index().drop('index',1).index.tolist())
interactions = interactions.drop('created_at',1)
sommati = interactions.groupby(['user_id','item_id']).aggregate(np.sum).reset_index().values
keys = zip(sommati[:,0],sommati[:,1])
fast_int = dict(zip(keys,sommati[:,2]))
fast_interacted = fast_int.keys()

items_fighi = items_fighi.as_matrix()

items = sps.csc_matrix(items)
itq = items.copy()
itq.data **= 2
norm = np.sqrt(itq.sum(axis=1))
norm = np.asarray(norm)[:,0]
inactive_indices = item_profiles[item_profiles['active_during_test'] == 0].index.values
norms = items / np.matrix(norm).T

def work():

    ratings = pd.DataFrame(columns=('user_id', 'item_id', 'rating'))
    a = 1

    for u in range(10000):
        user_id = users[u]
        interacted = interactions[interactions['user_id'] == user_id]['item_id'].values
        interacted = np.unique(interacted)
        indices = [ids_to_index.get(k) for k in interacted]

        similarities = np.dot(norms[indices],norms.T)
        similarities[:,indices] = 0 #metti a zero la similarità degli item con cui ho interagito
        similarities[:, inactive_indices] = 0 #remove inactive items (mmmmh, non dovrebbero esserci proprio)

        sum = np.sum(similarities,axis=0)
        sum = np.asarray(sum)[0]
        max = np.argsort(sum)[-100:][::-1]
        ratingss = sum[max] / (similarities.shape[0] if similarities.shape[0]>0 else 1)
        items_consigliati = [index_to_ids.get(k) for k in max]

        new = pd.DataFrame(np.asmatrix([np.repeat(user_id, 100), items_consigliati, ratingss]).T,columns=('user_id', 'item_id', 'rating'))
        ratings = ratings.append(new, ignore_index=True)


        if(u == (a*100)-1):
            print('done ' + str(u+1) + '/10000 users, users per second: ')
            a = a + 1


    ratings[['user_id', 'item_id']] = ratings[['user_id', 'item_id']].astype(int)
    return ratings

#popularity weights
def weight(i):
    return 1 - 1/i**(1/2)

def w(i):
    x = 0.01
    try:
        x = ws.get(i)
    except KeyError:
        x = 0.01
    return x

interactions = pd.read_csv('interactions.csv', delimiter="\t")
interactions = interactions.drop('created_at',1)
interactions = interactions[interactions['interaction_type'] == 1]
interactions = interactions.drop_duplicates()
interactions = interactions.groupby('item_id').aggregate(np.sum).reset_index().sort_values('interaction_type', ascending=False).drop('user_id',1)
ws = dict(zip(interactions['item_id'].values,np.apply_along_axis(weight,0,interactions['interaction_type'].values)))

def fill2(rec):
    suggestions = [0] * 10000
    for i in range(10000):
        suggestions[i] = rec[rec['user_id'] == users[i]].sort_values('rating',ascending=False).head(5)['item_id'].values
    return suggestions


#add top-pop to users without interactions
def fillpop(sugg):
    interactions = pd.read_csv('interactions.csv', delimiter="\t")
    most_popular = np.array([2778525,1244196,1386412,657183,2791339])
    a = np.unique(interactions['user_id'].values)
    anonimi = [u for u in users if u not in a]
    uds = target_users['user_id'].values
    uds_to_index = dict(zip(uds, range(uds.size)))

    indexeses = [uds_to_index.get(k) for k in anonimi]

    for i in indexeses:
        sugg[i] = most_popular

