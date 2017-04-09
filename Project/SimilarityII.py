import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from scipy.spatial import distance as d
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import project_utils as utils

#utility di sklearn che codifica attributi in forma di stringa, la usiamo per convertire le nazioni in numeri in modo coerente e confrontabile
le = preprocessing.LabelEncoder()

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
items = pd.read_csv("item_profile.csv", delimiter='\t')

items = items.fillna(0)
items['country'] = items['country'].replace(0,'null')
items['country'] = le.fit_transform(items['country'])

items = items[items['active_during_test'] == 1].reset_index().drop('index',1)
items = items.drop(['created_at', 'latitude', 'longitude', 'active_during_test'],1)


def myf(x):
    if isinstance(x,int):
        return str(x)
    else:
        return x

def myf2(s):
    return s.split(',')

corpus = items['tags'].ravel()
vectorizer = CountVectorizer(min_df=1)
corpus = list(map(myf,corpus))
corpus = list(map(myf2,corpus))
corpus = [item for sublist in corpus for item in sublist]
fdist = nltk.FreqDist(corpus)
common_tags = fdist.most_common(70)
common_tags = [tag[0] for tag in common_tags]
items['tags'] = items['tags'].astype(str)

for tag in common_tags:
    items['tag' + tag] = np.zeros(items.shape[0])

for index, row in items.iterrows():
    print(str(index))
    for tag in common_tags:
        if tag in row['tags']:
            items = items.set_value(index, 'tag'+tag, 1)

corpus2 = items['title'].ravel()
corpus2 = list(map(myf,corpus2))
corpus2 = list(map(myf2,corpus2))
corpus2 = [item for sublist in corpus2 for item in sublist]
fdist2 = nltk.FreqDist(corpus2)
common_titles = fdist.most_common(70)
common_titles = [title[0] for title in common_titles]
items['title'] = items['title'].astype(str)

for title in common_titles:
    items['title' + title] = np.zeros(items.shape[0])

for index, row in items.iterrows():
    print(str(index))
    for title in common_titles:
        if title in row['title']:
            items = items.set_value(index, 'title'+title, 1)

items = items.drop(['title','tags'],1)
to_encode = items.columns.difference(['id'])
items = utils.encode_feature(items,to_encode)
items = items.drop(to_encode,1)