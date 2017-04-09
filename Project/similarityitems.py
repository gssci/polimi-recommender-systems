import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from scipy.spatial import distance as d
import project_utils as util #output, similarity function and soon other things

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
items = items_r.fillna(0)
items['country'] = items['country'].replace(0,'null')
items['country'] = le.fit_transform(items['country'])
to_encode = ['discipline_id','industry_id','country','region']
util.encode_feature(items,to_encode)

items = items.drop(to_encode,1)
items = items.drop(['title','tags','latitude','longitude', 'created_at','active_during_test','id'],1)
items = items.as_matrix()

similarity_matrix = np.dot(items,np.transpose(items)) #MemoryError, quindi vaffanculo!