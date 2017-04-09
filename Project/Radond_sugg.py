import numpy as np
import Radond_sugg
import pandas as pd

users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1) 
items = pd.read_csv("item_profile.csv", usecols=['id','active_during_test'], delimiter="\t")
useful_items = items[items['active_during_test'] == 1]
useful_items_m = useful_items.as_matrix()
active_items = useful_items_m[:,0]

random_items = [0] * np.size(users)

for j in range(10000):
    tmp = [0] * 5

    for i in range(5):
        tmp[i] = Radond_sugg.choice(active_items)

    random_items[j] = tmp

result = np.column_stack((users,random_items))


#output to file
submission = open('submission.csv','w')
header = 'user_id,recommended_items'
submission.write(header + '\n')

for row in result:
    for i in range(np.size(row)):
        if i==0:
            line = str(row[0]) + ','
        elif i==5:
            line = line + str(row[i])
        else:
            line = line + str(row[i]) + ' '
    submission.write(line + '\n')

submission.close()